package com.demo;

import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.util.*;

import static org.apache.spark.sql.functions.*;

/**
 * 非核心埋点逐级补充算法 V2 —— 极致优化版。
 *
 * 相比 V1 的改进：
 *   1. 增量更新：维护 eventRemainingNewUV[]，每轮 O(|active|) 无需遍历用户集合
 *   2. 倒排索引 imei→eventIds (ArrayList)，选事件时只更新受影响的事件
 *   3. 每个非核心用户只在其首次被覆盖时处理一次
 *   4. Integer 事件 ID + 原生数组：long[]/String[] 替代 HashMap，消除装箱和哈希开销
 *   5. 紧凑活跃事件列表：int[] + swap-remove，跳过已归零事件
 *
 * 复杂度：
 *   baV1: O(steps × |E| × avg_users_per_event)
 *   V2: O(|users| × avg_events_per_user + steps × |active|)
 */
public class EventSupplementByMinPvEntry_V2 {

    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("必须提供 dt 和 bg 参数");
            System.exit(1);
        }

        String dt = args[0];
        String bg = args[1];
        int topN = args.length > 2 ? Integer.parseInt(args[2]) : Integer.MAX_VALUE;

        SparkSession spark = SparkSession.builder()
                .appName("EventSupplementByMinPvEntry_V2_" + dt + "_" + bg)
                .enableHiveSupport()
                .config("hive.exec.dynamic.partition.mode", "nonstrict")
                .getOrCreate();

        String sourceTable = "hdp_teu_dpd_wx_flow.jf_event_impact_daily_v2";
        String targetTable = "hdp_teu_dpd_wx_flow.jf_event_supplement_by_pv_rank_v2";

        // ========== 1. 读取源表 ==========
        Dataset<Row> sourceDF = spark.table(sourceTable)
                .filter(col("dt").equalTo(dt).and(col("bg").equalTo(bg)))
                .cache();

        // ========== 2. 全局指标 ==========
        Row globalRow = sourceDF.filter(col("rank").equalTo(0)).head();
        long totalUV = globalRow.getLong(globalRow.fieldIndex("total_uv"));
        long totalPV = globalRow.getLong(globalRow.fieldIndex("total_pv"));
        long coreUV  = globalRow.getLong(globalRow.fieldIndex("core_uv"));
        long corePV  = globalRow.getLong(globalRow.fieldIndex("core_pv"));

        System.out.printf("总UV=%d, 总PV=%d, 核心UV=%d, 核心PV=%d%n",
                totalUV, totalPV, coreUV, corePV);

        // ========== 3. 展开非核心埋点用户映射 ==========
        Dataset<Row> expandedDF = sourceDF
                .filter(col("rank").gt(0))
                .withColumn("event_key", concat(col("pagetype"), lit("|"), col("actiontype")))
                .select(col("event_key"), col("event_pv"),
                        explode(split(col("imei_info_list"), ",")).as("info"))
                .select(col("event_key"), col("event_pv"),
                        split(col("info"), ":").getItem(0).as("imei"))
                .filter(col("imei").isNotNull().and(length(col("imei")).gt(0)))
                .cache();

        // ========== 4. 核心用户列表 ==========
        Dataset<Row> coreUserDF = sourceDF.filter(col("rank").equalTo(-1))
                .select(explode(split(col("imei_info_list"), ",")).as("info"))
                .select(split(col("info"), ":").getItem(0).as("imei"))
                .filter(col("imei").isNotNull().and(length(col("imei")).gt(0)))
                .distinct();

        // 排除核心用户，只保留非核心用户的 (event_key, imei) 对
        Dataset<Row> nonCoreDF = expandedDF
                .join(broadcast(coreUserDF), expandedDF.col("imei").equalTo(coreUserDF.col("imei")), "left_anti")
                .cache();

        // ========== 5. Collect 到 Driver 构建索引 ==========
        List<Row> eventMeta = sourceDF
                .filter(col("rank").gt(0))
                .select(col("pagetype"), col("actiontype"), col("event_pv"))
                .distinct()
                .collectAsList();

        List<Row> userPairs = nonCoreDF
                .select("event_key", "imei")
                .distinct()
                .collectAsList();

        // ---- 阶段一：构建 String→int 事件ID映射 + 元数据数组 ----
        Map<String, Integer> eventKeyToId = new HashMap<>();
        List<String> eventKeyList = new ArrayList<>();

        for (Row r : eventMeta) {
            String pagetype = r.getString(0);
            String actiontype = r.getString(1);
            String key = pagetype + "|" + actiontype;
            if (!eventKeyToId.containsKey(key)) {
                eventKeyToId.put(key, eventKeyList.size());
                eventKeyList.add(key);
            }
        }

        int numEvents = eventKeyList.size();
        String[] eventPagetypes = new String[numEvents];
        String[] eventActiontypes = new String[numEvents];
        long[] eventPVs = new long[numEvents];

        for (Row r : eventMeta) {
            String pagetype = r.getString(0);
            String actiontype = r.getString(1);
            long pv = r.getLong(2);
            String key = pagetype + "|" + actiontype;
            int id = eventKeyToId.get(key);
            eventPagetypes[id] = pagetype;
            eventActiontypes[id] = actiontype;
            eventPVs[id] = pv;
        }

        // ---- 阶段二：构建 eventUsersSets 和倒排索引（用 ArrayList，pair已distinct无需去重）----
        @SuppressWarnings("unchecked")
        Set<String>[] eventUsersSets = new Set[numEvents];
        for (int i = 0; i < numEvents; i++) {
            eventUsersSets[i] = new HashSet<>();
        }

        Map<String, List<Integer>> imeiToEventIds = new HashMap<>();

        for (Row r : userPairs) {
            String key = r.getString(0);
            String imei = r.getString(1);
            Integer eventId = eventKeyToId.get(key);
            if (eventId == null) continue;

            eventUsersSets[eventId].add(imei);

            List<Integer> list = imeiToEventIds.get(imei);
            if (list == null) {
                list = new ArrayList<>(2); // 多数用户只出现在1-2个事件中
                imeiToEventIds.put(imei, list);
            }
            list.add(eventId);
        }

        // ---- 阶段三：初始化剩余UV计数 和 活跃事件列表 ----
        long[] eventRemainingNewUV = new long[numEvents];
        int[] activeEventIds = new int[numEvents];
        int activeCount = 0;

        for (int i = 0; i < numEvents; i++) {
            long uv = eventUsersSets[i].size();
            eventRemainingNewUV[i] = uv;
            if (uv > 0) {
                activeEventIds[activeCount++] = i;
            }
        }

        // 释放中间 HashMap，减少内存压力
        eventKeyToId = null;

        Set<String> covered = new HashSet<>();

        // ========== 6. Driver 端贪心选择 ==========
        List<Map<String, Object>> stepRows = new ArrayList<>();

        // Step 0：仅核心事件基线
        Map<String, Object> step0 = new LinkedHashMap<>();
        step0.put("supplement_step", 0);
        step0.put("supplemented_pagetype", "");
        step0.put("supplemented_actiontype", "");
        step0.put("supplemented_event_pv", 0L);
        step0.put("new_uv_gained", 0L);
        step0.put("cum_uv", coreUV);
        step0.put("cum_pv", corePV);
        step0.put("total_uv", totalUV);
        step0.put("total_pv", totalPV);
        step0.put("core_uv", coreUV);
        step0.put("core_pv", corePV);
        step0.put("uv_coverage_rate", (double) coreUV / totalUV);
        step0.put("pv_growth_rate", 0.0);
        stepRows.add(step0);

        long cumUV = coreUV;
        long cumPV = corePV;
        int step = 0;

        System.out.printf("Step %3d | %-8s %-20s %-25s | PV=+%8d | 新UV=+%8d | 累积UV=%10d (%.2f%%) | 累积PV=%15d (%6.2f%%)%n",
                0, "[基线]", "", "",
                0, 0,
                cumUV, (double) cumUV / totalUV * 100,
                cumPV, (double) (cumPV - corePV) / totalPV * 100);

        while (activeCount > 0 && step < topN) {
            // 紧凑扫描活跃事件列表，同时找最优 + 清理归零事件
            int bestId = -1;
            int bestIdx = -1;
            double bestScore = -1;
            long bestNewUV = 0;

            for (int i = 0; i < activeCount; ) {
                int eventId = activeEventIds[i];
                long newUV = eventRemainingNewUV[eventId];
                if (newUV == 0) {
                    // swap-remove：与末尾交换并缩容
                    activeEventIds[i] = activeEventIds[--activeCount];
                    continue;
                }
                double score = (double) newUV / eventPVs[eventId];
                if (score > bestScore) {
                    bestScore = score;
                    bestId = eventId;
                    bestIdx = i;
                    bestNewUV = newUV;
                }
                i++;
            }

            if (bestId == -1) break;

            // 选中该事件，swap-remove 从活跃列表移除
            step++;
            activeEventIds[bestIdx] = activeEventIds[--activeCount];
            eventRemainingNewUV[bestId] = 0;

            // 增量更新：只处理此事件中首次被覆盖的用户
            for (String u : eventUsersSets[bestId]) {
                if (!covered.add(u)) continue; // 已覆盖，跳过

                List<Integer> affectedEvents = imeiToEventIds.get(u);
                if (affectedEvents != null) {
                    for (int affectedId : affectedEvents) {
                        eventRemainingNewUV[affectedId]--;
                    }
                }
            }

            cumUV += bestNewUV;
            cumPV += eventPVs[bestId];

            Map<String, Object> row = new LinkedHashMap<>();
            row.put("supplement_step", step);
            row.put("supplemented_pagetype", eventPagetypes[bestId]);
            row.put("supplemented_actiontype", eventActiontypes[bestId]);
            row.put("supplemented_event_pv", eventPVs[bestId]);
            row.put("new_uv_gained", bestNewUV);
            row.put("cum_uv", cumUV);
            row.put("cum_pv", cumPV);
            row.put("total_uv", totalUV);
            row.put("total_pv", totalPV);
            row.put("core_uv", coreUV);
            row.put("core_pv", corePV);
            row.put("uv_coverage_rate", (double) cumUV / totalUV);
            row.put("pv_growth_rate", (double) (cumPV - corePV) / totalPV);
            stepRows.add(row);

            System.out.printf("Step %3d | %-8s %-20s %-25s | PV=+%8d | 新UV=+%8d | 累积UV=%10d (%.2f%%) | 累积PV=%15d (%6.2f%%)%n",
                    step, "[补充]", eventPagetypes[bestId], eventActiontypes[bestId],
                    eventPVs[bestId], bestNewUV,
                    cumUV, (double) cumUV / totalUV * 100,
                    cumPV, (double) (cumPV - corePV) / totalPV * 100);
        }

        // ========== 7. 汇总 ==========
        System.out.printf("%n========================================%n" +
                        "补充完成：%d 步%n" +
                        "  基线 UV=%d (%.2f%%)  PV=%d%n" +
                        "  最终 UV=%d (%.2f%%)  PV=%d (增 %.2f%%)%n" +
                        "  覆盖率提升 %.2f%% → %.2f%%%n" +
                        "========================================%n",
                step,
                coreUV, (double) coreUV / totalUV * 100, corePV,
                cumUV, (double) cumUV / totalUV * 100, cumPV, (double) (cumPV - corePV) / totalPV * 100,
                (double) coreUV / totalUV * 100, (double) cumUV / totalUV * 100
        );

        // ========== 8. 写回 Hive ==========
        StructType outputSchema = new StructType()
                .add("supplement_step",           DataTypes.IntegerType)
                .add("supplemented_pagetype",     DataTypes.StringType)
                .add("supplemented_actiontype",   DataTypes.StringType)
                .add("supplemented_event_pv",     DataTypes.LongType)
                .add("new_uv_gained",             DataTypes.LongType)
                .add("cum_uv",                    DataTypes.LongType)
                .add("cum_pv",                    DataTypes.LongType)
                .add("total_uv",                  DataTypes.LongType)
                .add("total_pv",                  DataTypes.LongType)
                .add("core_uv",                   DataTypes.LongType)
                .add("core_pv",                   DataTypes.LongType)
                .add("uv_coverage_rate",          DataTypes.DoubleType)
                .add("pv_growth_rate",            DataTypes.DoubleType);

        List<Row> sparkRows = new ArrayList<>();
        for (Map<String, Object> row : stepRows) {
            sparkRows.add(RowFactory.create(
                    row.get("supplement_step"),
                    row.get("supplemented_pagetype"),
                    row.get("supplemented_actiontype"),
                    row.get("supplemented_event_pv"),
                    row.get("new_uv_gained"),
                    row.get("cum_uv"),
                    row.get("cum_pv"),
                    row.get("total_uv"),
                    row.get("total_pv"),
                    row.get("core_uv"),
                    row.get("core_pv"),
                    row.get("uv_coverage_rate"),
                    row.get("pv_growth_rate")
            ));
        }

        Dataset<Row> resultDF = spark.createDataFrame(sparkRows, outputSchema);
        resultDF.createOrReplaceTempView("tmp_supplement_result");
        spark.sql(
                "INSERT OVERWRITE TABLE " + targetTable +
                        " PARTITION(dt='" + dt + "', bg='" + bg + "')" +
                        " SELECT supplement_step, supplemented_pagetype, supplemented_actiontype," +
                        "        supplemented_event_pv, new_uv_gained," +
                        "        cum_uv, cum_pv, total_uv, total_pv," +
                        "        core_uv, core_pv," +
                        "        uv_coverage_rate, pv_growth_rate" +
                        " FROM tmp_supplement_result"
        );

        expandedDF.unpersist();
        nonCoreDF.unpersist();
        sourceDF.unpersist();

        System.out.println("结果已写入 " + targetTable);
        spark.stop();
    }
}
