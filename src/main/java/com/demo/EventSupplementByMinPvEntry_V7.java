package com.demo;

import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.util.*;

import static org.apache.spark.sql.functions.*;

/**
 * 非核心埋点逐级补充算法 V7 —— PV 预算约束下的最大覆盖（Budgeted Maximum Coverage）。
 *
 * 相比 V5/V6 改动：
 *   1. 停止条件由固定步数(topN)改为 PV 预算上限，贪心保证不低于最优解 63%
 *   2. 排序公式回到 nUV/PV（性价比），由预算约束全局控制 PV 膨胀
 *   3. 第三个参数 pvBudgetRatio：目标总 PV / 核心 PV 的倍数，默认 1.5
 */
public class EventSupplementByMinPvEntry_V7 {

    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("必须提供 dt 和 bg 参数");
            System.exit(1);
        }

        String dt = args[0];
        String bg = args[1];
        double pvBudgetRatio = args.length > 2 ? Double.parseDouble(args[2]) : 1.5;

        SparkSession spark = SparkSession.builder()
                .appName("EventSupplementByMinPvEntry_V7_" + dt + "_" + bg)
                .enableHiveSupport()
                .config("hive.exec.dynamic.partition.mode", "nonstrict")
                .getOrCreate();

        spark.conf().set("spark.sql.shuffle.partitions", "800");
        spark.conf().set("spark.sql.adaptive.enabled", "true");
        spark.conf().set("spark.sql.adaptive.coalescePartitions.enabled", "true");

        String sourceTable = "hdp_teu_dpd_wx_flow.jf_event_impact_daily_v4";
        String targetTable = "hdp_teu_dpd_wx_flow.jf_event_supplement_by_pv_rank_v5";

        // ========== 1. 读取源表，按 imei 重分布后缓存并立即物化 ==========
        Dataset<Row> sourceDF = spark.table(sourceTable)
                .filter(col("dt").equalTo(dt).and(col("bg").equalTo(bg)))
                .repartition(800, col("imei"))
                .cache();
        sourceDF.count();

        // ========== 2. 全局 + 核心指标（一次 agg，省一次全表扫描）==========
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

        long pvBudget = (long) (corePV * pvBudgetRatio);

        System.out.printf("总UV=%d 总PV=%d 核心UV=%d 核心PV=%d PV预算=%d (%.2fx)%n",
                totalUV, totalPV, coreUV, corePV, pvBudget, pvBudgetRatio);

        // ========== 3. 核心用户列表 ==========
        Dataset<Row> coreUserDF = sourceDF
                .filter(col("is_core").equalTo(1))
                .select("imei").distinct();

        // ========== 4. 非核心用户数据 ==========
        Dataset<Row> nonCoreDF = sourceDF
                .filter(col("is_core").equalTo(0))
                .select(col("pagetype"), col("actiontype"), col("imei"), col("pv"))
                .cache();

        long nonCoreUserEstimate = totalUV - coreUV;

        Dataset<Row> pureNonCoreDF = nonCoreDF
                .join(broadcast(coreUserDF), nonCoreDF.col("imei").equalTo(coreUserDF.col("imei")), "left_anti")
                .cache();

        // ========== 5. Collect 到 Driver ==========
        List<Row> eventMeta = sourceDF
                .groupBy(col("pagetype"), col("actiontype"))
                .agg(sum("pv").as("event_pv"))
                .collectAsList();

        List<Row> userPairs = pureNonCoreDF
                .select("pagetype", "actiontype", "imei")
                .distinct()
                .collectAsList();

        System.out.printf("事件数=%d 用户-事件对=%d%n", eventMeta.size(), userPairs.size());

        // ---- 阶段一：事件ID映射 + 元数据数组（按 index 直接赋值）----
        int numEvents = eventMeta.size();
        Map<String, Integer> eventKeyToId = new HashMap<>(numEvents * 2);
        String[] eventPagetypes = new String[numEvents];
        String[] eventActiontypes = new String[numEvents];
        long[] eventPVs = new long[numEvents];

        for (int i = 0; i < numEvents; i++) {
            Row r = eventMeta.get(i);
            String pagetype = r.getString(0);
            String actiontype = r.getString(1);
            eventKeyToId.put(pagetype + "|" + actiontype, i);
            eventPagetypes[i] = pagetype;
            eventActiontypes[i] = actiontype;
            eventPVs[i] = r.getLong(2);
        }
        eventMeta = null;

        // ---- 阶段二：构建 eventUsersSets 和倒排索引 ----
        @SuppressWarnings("unchecked")
        Set<String>[] eventUsersSets = new Set[numEvents];
        for (int i = 0; i < numEvents; i++) {
            eventUsersSets[i] = new HashSet<>();
        }

        Map<String, List<Integer>> imeiToEventIds = new HashMap<>((int) (nonCoreUserEstimate * 1.5));

        for (Row r : userPairs) {
            String key = r.getString(0) + "|" + r.getString(1);
            Integer eventId = eventKeyToId.get(key);
            if (eventId == null) continue;

            String imei = r.getString(2);
            eventUsersSets[eventId].add(imei);

            List<Integer> list = imeiToEventIds.get(imei);
            if (list == null) {
                list = new ArrayList<>(2);
                imeiToEventIds.put(imei, list);
            }
            list.add(eventId);
        }
        userPairs = null;

        // ---- 阶段三：初始化计数器 ----
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

        eventKeyToId = null;
        Set<String> covered = new HashSet<>((int) (nonCoreUserEstimate * 1.5));

        // ========== 6. 贪心选择（PV 预算约束）==========
        List<Map<String, Object>> stepRows = new ArrayList<>();

        // Step 0 基线
        Map<String, Object> s0 = new LinkedHashMap<>(8);
        s0.put("supplement_step", 0);
        s0.put("supplemented_pagetype", "");
        s0.put("supplemented_actiontype", "");
        s0.put("supplemented_event_pv", 0L);
        s0.put("new_uv_gained", 0L);
        s0.put("cum_uv", coreUV);
        s0.put("cum_pv", corePV);
        stepRows.add(s0);

        long cumUV = coreUV;
        long cumPV = corePV;
        int step = 0;

        System.out.printf("Step %4d | %-20s %-25s | PV=+%10d | UV=+%8d | cumUV=%10d | cumPV=%15d | 预算剩余=%10d%n",
                0, "[基线]", "", 0, 0, cumUV, cumPV, pvBudget - cumPV);

        while (activeCount > 0) {
            int bestId = -1;
            int bestIdx = -1;
            double bestScore = -1;
            long bestNewUV = 0;
            long bestPV = 0;

            long remaining = pvBudget - cumPV;

            for (int i = 0; i < activeCount; ) {
                int eid = activeEventIds[i];
                long nUV = eventRemainingNewUV[eid];
                if (nUV == 0) {
                    activeEventIds[i] = activeEventIds[--activeCount];
                    continue;
                }
                // 只考虑 PV 不超预算的事件
                if (eventPVs[eid] > remaining) {
                    i++;
                    continue;
                }
                double score = (double) nUV / eventPVs[eid];
                if (score > bestScore) {
                    bestScore = score;
                    bestId = eid;
                    bestIdx = i;
                    bestNewUV = nUV;
                    bestPV = eventPVs[eid];
                }
                i++;
            }

            if (bestId == -1) break;

            step++;
            activeEventIds[bestIdx] = activeEventIds[--activeCount];

            // 增量更新
            for (String u : eventUsersSets[bestId]) {
                if (!covered.add(u)) continue;
                List<Integer> affected = imeiToEventIds.get(u);
                if (affected != null) {
                    for (int aid : affected) {
                        eventRemainingNewUV[aid]--;
                    }
                }
            }

            cumUV += bestNewUV;
            cumPV += bestPV;

            Map<String, Object> row = new LinkedHashMap<>(8);
            row.put("supplement_step", step);
            row.put("supplemented_pagetype", eventPagetypes[bestId]);
            row.put("supplemented_actiontype", eventActiontypes[bestId]);
            row.put("supplemented_event_pv", bestPV);
            row.put("new_uv_gained", bestNewUV);
            row.put("cum_uv", cumUV);
            row.put("cum_pv", cumPV);
            stepRows.add(row);

            System.out.printf("Step %4d | %-20s %-25s | PV=+%10d | UV=+%8d | cumUV=%10d | cumPV=%15d | 预算剩余=%10d%n",
                    step, eventPagetypes[bestId], eventActiontypes[bestId],
                    bestPV, bestNewUV, cumUV, cumPV, pvBudget - cumPV);
        }

        System.out.printf("%n===== 补充完成：%d 步 | UV %d→%d | PV %d→%d (预算=%d, 使用率=%.1f%%) =====%n",
                step, coreUV, cumUV, corePV, cumPV, pvBudget,
                100.0 * (cumPV - corePV) / (pvBudget - corePV));

        // ========== 7. 写回 Hive（仅 7 列）==========
        StructType outputSchema = new StructType()
                .add("supplement_step",           DataTypes.IntegerType)
                .add("supplemented_pagetype",     DataTypes.StringType)
                .add("supplemented_actiontype",   DataTypes.StringType)
                .add("supplemented_event_pv",     DataTypes.LongType)
                .add("new_uv_gained",             DataTypes.LongType)
                .add("cum_uv",                    DataTypes.LongType)
                .add("cum_pv",                    DataTypes.LongType);

        List<Row> sparkRows = new ArrayList<>(stepRows.size());
        for (Map<String, Object> row : stepRows) {
            sparkRows.add(RowFactory.create(
                    row.get("supplement_step"),
                    row.get("supplemented_pagetype"),
                    row.get("supplemented_actiontype"),
                    row.get("supplemented_event_pv"),
                    row.get("new_uv_gained"),
                    row.get("cum_uv"),
                    row.get("cum_pv")
            ));
        }

        Dataset<Row> resultDF = spark.createDataFrame(sparkRows, outputSchema);
        resultDF.createOrReplaceTempView("tmp_supplement_result");
        spark.sql(
                "INSERT OVERWRITE TABLE " + targetTable +
                        " PARTITION(dt='" + dt + "', bg='" + bg + "')" +
                        " SELECT supplement_step, supplemented_pagetype, supplemented_actiontype," +
                        "        supplemented_event_pv, new_uv_gained," +
                        "        cum_uv, cum_pv" +
                        " FROM tmp_supplement_result"
        );

        nonCoreDF.unpersist();
        pureNonCoreDF.unpersist();
        sourceDF.unpersist();

        System.out.println("结果已写入 " + targetTable);
        spark.stop();
    }
}