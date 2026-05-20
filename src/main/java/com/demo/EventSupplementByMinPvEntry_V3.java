package com.demo;

import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.util.*;

import static org.apache.spark.sql.functions.*;

/**
 * 非核心埋点逐级补充算法 V3 —— 适配扁平源表 jf_event_impact_daily_v3（极致并行优化版）。
 *
 * 源表字段：imei, pagetype, actiontype, is_core(1=核心/0=非核心), pv
 *
 * 并行优化点：
 *   1. 全局+核心指标合并为一次 agg 扫描（减少一次全表 Shuffle）
 *   2. sourceDF 按 imei 重分布，提升 groupBy/distinct/join 并行度
 *   3. 所有 HashMap 预分配容量，避免扩容
 *   4. Spark shuffle 分区数设为 executor core 的 3-4 倍
 */
public class EventSupplementByMinPvEntry_V3 {

    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("必须提供 dt 和 bg 参数");
            System.exit(1);
        }

        String dt = args[0];
        String bg = args[1];
        int topN = args.length > 2 ? Integer.parseInt(args[2]) : Integer.MAX_VALUE;

        SparkSession spark = SparkSession.builder()
                .appName("EventSupplementByMinPvEntry_V3_" + dt + "_" + bg)
                .enableHiveSupport()
                .config("hive.exec.dynamic.partition.mode", "nonstrict")
                .getOrCreate();

        // ---- 并行度设置 ----
        spark.conf().set("spark.sql.shuffle.partitions", "800");
        spark.conf().set("spark.sql.adaptive.enabled", "true");
        spark.conf().set("spark.sql.adaptive.coalescePartitions.enabled", "true");

        String sourceTable = "hdp_teu_dpd_wx_flow.jf_event_impact_daily_v3";
        String targetTable = "hdp_teu_dpd_wx_flow.jf_event_supplement_by_pv_rank_v3";

        // ========== 1. 读取源表并按 imei 重分布（提升后续 groupBy/distinct/join 并行度）==========
        Dataset<Row> sourceDF = spark.table(sourceTable)
                .filter(col("dt").equalTo(dt).and(col("bg").equalTo(bg)))
                .repartition(800, col("imei"))
                .cache();

        // 触发缓存物化，后续所有操作从内存读取
        sourceDF.count();

        // ========== 2. 全局 + 核心指标（合并为一次 agg，减少一次全表扫描）==========
        Row metrics = sourceDF.agg(
                countDistinct("imei").as("total_uv"),
                sum("pv").as("total_pv"),
                countDistinct(when(col("is_core").equalTo(1), col("imei"))).as("core_uv"),
                sum(when(col("is_core").equalTo(1), col("pv")).otherwise(0L)).as("core_pv")
        ).head();

        long totalUV = metrics.getLong(0);
        long totalPV = metrics.getLong(1);
        long coreUV  = metrics.getLong(2);
        long corePV  = metrics.getLong(3);

        System.out.printf("总UV=%d, 总PV=%d, 核心UV=%d, 核心PV=%d%n",
                totalUV, totalPV, coreUV, corePV);

        // ========== 3. 核心用户列表 ==========
        Dataset<Row> coreUserDF = sourceDF
                .filter(col("is_core").equalTo(1))
                .select("imei")
                .distinct();

        // ========== 4. 非核心用户数据 ==========
        Dataset<Row> nonCoreDF = sourceDF
                .filter(col("is_core").equalTo(0))
                .select(col("pagetype"), col("actiontype"), col("imei"), col("pv"))
                .cache();

        long nonCoreUserEstimate = totalUV - coreUV;
        System.out.printf("预估非核心用户数: %d%n", nonCoreUserEstimate);

        // 排除核心用户
        Dataset<Row> pureNonCoreDF = nonCoreDF
                .join(broadcast(coreUserDF), nonCoreDF.col("imei").equalTo(coreUserDF.col("imei")), "left_anti")
                .cache();

        // ========== 5. Collect 到 Driver 构建索引 ==========
        List<Row> eventMeta = sourceDF
                .groupBy(col("pagetype"), col("actiontype"))
                .agg(sum("pv").as("event_pv"))
                .collectAsList();

        List<Row> userPairs = pureNonCoreDF
                .select("pagetype", "actiontype", "imei")
                .distinct()
                .collectAsList();

        System.out.printf("事件数: %d, 用户-事件对数: %d%n", eventMeta.size(), userPairs.size());

        // ---- 阶段一：构建 String→int 事件ID映射 + 元数据数组（预分配容量）----
        int numEvents = eventMeta.size();
        Map<String, Integer> eventKeyToId = new HashMap<>(numEvents * 2);
        String[] eventPagetypes = new String[numEvents];
        String[] eventActiontypes = new String[numEvents];
        long[] eventPVs = new long[numEvents];

        for (Row r : eventMeta) {
            String pagetype = r.getString(0);
            String actiontype = r.getString(1);
            long pv = r.getLong(2);
            String key = pagetype + "|" + actiontype;
            Integer existing = eventKeyToId.get(key);
            if (existing == null) {
                int id = eventKeyToId.size();
                eventKeyToId.put(key, id);
                eventPagetypes[id] = pagetype;
                eventActiontypes[id] = actiontype;
                eventPVs[id] = pv;
            }
        }

        // ---- 阶段二：构建 eventUsersSets 和倒排索引 ----
        int actualNumEvents = eventKeyToId.size();
        @SuppressWarnings("unchecked")
        Set<String>[] eventUsersSets = new Set[actualNumEvents];
        for (int i = 0; i < actualNumEvents; i++) {
            eventUsersSets[i] = new HashSet<>();
        }

        // 预分配倒排索引容量（非核心用户数 × 2，每个用户平均 ~2 个事件）
        Map<String, List<Integer>> imeiToEventIds = new HashMap<>((int) (nonCoreUserEstimate * 1.5));

        for (Row r : userPairs) {
            String pagetype = r.getString(0);
            String actiontype = r.getString(1);
            String imei = r.getString(2);
            String key = pagetype + "|" + actiontype;
            Integer eventId = eventKeyToId.get(key);
            if (eventId == null) continue;

            eventUsersSets[eventId].add(imei);

            List<Integer> list = imeiToEventIds.get(imei);
            if (list == null) {
                list = new ArrayList<>(2);
                imeiToEventIds.put(imei, list);
            }
            list.add(eventId);
        }

        // ---- 阶段三：初始化剩余UV计数 和 活跃事件列表 ----
        long[] eventRemainingNewUV = new long[actualNumEvents];
        int[] activeEventIds = new int[actualNumEvents];
        int activeCount = 0;

        for (int i = 0; i < actualNumEvents; i++) {
            long uv = eventUsersSets[i].size();
            eventRemainingNewUV[i] = uv;
            if (uv > 0) {
                activeEventIds[activeCount++] = i;
            }
        }

        // 释放中间引用
        eventKeyToId = null;

        Set<String> covered = new HashSet<>((int) (nonCoreUserEstimate * 1.5));

        // ========== 6. Driver 端贪心选择 ==========
        List<Map<String, Object>> stepRows = new ArrayList<>();

        // Step 0：核心用户基线
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
            int bestId = -1;
            int bestIdx = -1;
            double bestScore = -1;
            long bestNewUV = 0;

            for (int i = 0; i < activeCount; ) {
                int eventId = activeEventIds[i];
                long newUV = eventRemainingNewUV[eventId];
                if (newUV == 0) {
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

            step++;
            activeEventIds[bestIdx] = activeEventIds[--activeCount];
            eventRemainingNewUV[bestId] = 0;

            for (String u : eventUsersSets[bestId]) {
                if (!covered.add(u)) continue;

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

        nonCoreDF.unpersist();
        pureNonCoreDF.unpersist();
        sourceDF.unpersist();

        System.out.println("结果已写入 " + targetTable);
        spark.stop();
    }
}
