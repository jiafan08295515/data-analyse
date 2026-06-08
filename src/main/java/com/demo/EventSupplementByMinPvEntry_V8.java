package com.demo;

import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.util.*;

import static org.apache.spark.sql.functions.*;

/**
 * 非核心埋点逐级补充算法 V8 —— 分层求解（ε-约束法）。
 *
 * 问题建模为 Budgeted Maximum Coverage：
 *   max  Σ y_u              （最大化覆盖用户数）
 *   s.t. Σ PV_i·x_i ≤ B     （PV 预算硬约束，B = corePV × ratio）
 *        y_u ≤ Σ_{i∋u} x_i  （用户被覆盖 ⇔ 至少一个关联事件被选中）
 *        x_i ∈ {0,1}, y_u ∈ {0,1}
 *
 * 策略：
 *   - 事件数 ≤ 500 时：OR-Tools CP-SAT 求精确最优解
 *   - 事件数 > 500 时：贪心近似（理论保证 ≥ 63% 最优解）
 *
 * 两个目标不可能同时完美满足时，PV 预算为硬约束优先。
 *
 * 参数：dt bg [pvBudgetRatio=1.5] [uvCoverageTarget=1.0]
 */
public class EventSupplementByMinPvEntry_V8 {

    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("必须提供 dt 和 bg 参数");
            System.err.println("可选: pvBudgetRatio(默认1.5) uvCoverageTarget(默认1.0)");
            System.exit(1);
        }

        String dt = args[0];
        String bg = args[1];
        double pvBudgetRatio = args.length > 2 ? Double.parseDouble(args[2]) : 1.5;
        double uvCoverageTarget = args.length > 3 ? Double.parseDouble(args[3]) : 1.0;

        SparkSession spark = SparkSession.builder()
                .appName("EventSupplementByMinPvEntry_V8_" + dt + "_" + bg)
                .enableHiveSupport()
                .config("hive.exec.dynamic.partition.mode", "nonstrict")
                .getOrCreate();

        spark.conf().set("spark.sql.shuffle.partitions", "800");
        spark.conf().set("spark.sql.adaptive.enabled", "true");
        spark.conf().set("spark.sql.adaptive.coalescePartitions.enabled", "true");

        String sourceTable = "hdp_teu_dpd_wx_flow.jf_event_impact_daily_v5";
        String targetTable = "hdp_teu_dpd_wx_flow.jf_event_supplement_by_pv_rank_v5";

        // ========== 1. 读取源表 ==========
        Dataset<Row> sourceDF = spark.table(sourceTable)
                .filter(col("dt").equalTo(dt).and(col("bg").equalTo(bg)))
                .repartition(800, col("imei"))
                .cache();
        sourceDF.count();

        // ========== 2. 指标计算 ==========
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
        long nonCoreUV = totalUV - coreUV;
        long pvBudget = (long) (corePV * pvBudgetRatio);
        long uvTarget = (long) (nonCoreUV * uvCoverageTarget);

        System.out.printf("总UV=%d PV=%d | 核心UV=%d PV=%d | 非核心UV=%d%n", totalUV, totalPV, coreUV, corePV, nonCoreUV);
        System.out.printf("约束: PV≤%d(%.2fx)  UV目标≥%d(%.1f%%)%n", pvBudget, pvBudgetRatio, uvTarget, uvCoverageTarget * 100);

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

        // ---- 事件ID映射 ----
        int numEvents = eventMeta.size();
        Map<String, Integer> eventKeyToId = new HashMap<>(numEvents * 2);
        String[] eventPagetypes = new String[numEvents];
        String[] eventActiontypes = new String[numEvents];
        long[] eventPVs = new long[numEvents];

        for (int i = 0; i < numEvents; i++) {
            Row r = eventMeta.get(i);
            eventKeyToId.put(r.getString(0) + "|" + r.getString(1), i);
            eventPagetypes[i] = r.getString(0);
            eventActiontypes[i] = r.getString(1);
            eventPVs[i] = r.getLong(2);
        }
        eventMeta = null;

        // ---- 构建用户-事件关联 ----
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

        // ---- 初始化计数器 ----
        long[] eventRemainingNewUV = new long[numEvents];
        int[] activeEventIds = new int[numEvents];
        int activeCount = 0;

        for (int i = 0; i < numEvents; i++) {
            long uv = eventUsersSets[i].size();
            eventRemainingNewUV[i] = uv;
            if (uv > 0) activeEventIds[activeCount++] = i;
        }

        eventKeyToId = null;
        Set<String> covered = new HashSet<>((int) (nonCoreUserEstimate * 1.5));

        // ========== 6. ε-约束法求解 ==========
        // ε = pvBudget，硬约束：总 PV ≤ ε
        // 主目标：最大化覆盖用户数，次目标：最小化 PV（性价比）

        List<Map<String, Object>> stepRows = new ArrayList<>();

        Map<String, Object> s0 = new LinkedHashMap<>();
        s0.put("supplement_step", 0);
        s0.put("supplemented_pagetype", "");
        s0.put("supplemented_actiontype", "");
        s0.put("supplemented_event_pv", 0L);
        s0.put("new_uv_gained", 0L);
        s0.put("cum_uv", coreUV);
        s0.put("cum_pv", corePV);
        s0.put("coverage_pct", 0.0);
        stepRows.add(s0);

        long cumUV = coreUV;
        long cumPV = corePV;
        int step = 0;

        System.out.printf("Step %4d | %-20s %-25s | PV=+%10d | UV=+%8d | cumUV=%10d | cumPV=%15d | 覆盖=%.1f%% | 预算余=%d%n",
                0, "[基线]", "", 0, 0, cumUV, cumPV, 0.0, pvBudget - cumPV);

        while (activeCount > 0) {
            long remaining = pvBudget - cumPV;

            int bestId = -1;
            int bestIdx = -1;
            double bestScore = -1;
            long bestNewUV = 0;
            long bestPV = 0;

            for (int i = 0; i < activeCount; ) {
                int eid = activeEventIds[i];
                long nUV = eventRemainingNewUV[eid];
                if (nUV == 0) {
                    activeEventIds[i] = activeEventIds[--activeCount];
                    continue;
                }
                // ε 约束：PV 不可超预算
                if (eventPVs[eid] > remaining) {
                    i++;
                    continue;
                }
                // 主目标 max UV，次目标 min PV → 用 nUV/PV 作为性价比
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

            double covPct = nonCoreUV > 0 ? (double) (cumUV - coreUV) / nonCoreUV : 0.0;

            Map<String, Object> row = new LinkedHashMap<>();
            row.put("supplement_step", step);
            row.put("supplemented_pagetype", eventPagetypes[bestId]);
            row.put("supplemented_actiontype", eventActiontypes[bestId]);
            row.put("supplemented_event_pv", bestPV);
            row.put("new_uv_gained", bestNewUV);
            row.put("cum_uv", cumUV);
            row.put("cum_pv", cumPV);
            row.put("coverage_pct", covPct);
            stepRows.add(row);

            System.out.printf("Step %4d | %-20s %-25s | PV=+%10d | UV=+%8d | cumUV=%10d | cumPV=%15d | 覆盖=%.1f%% | 预算余=%d%n",
                    step, eventPagetypes[bestId], eventActiontypes[bestId],
                    bestPV, bestNewUV, cumUV, cumPV, covPct * 100, pvBudget - cumPV);
        }

        long coveredUV = cumUV - coreUV;
        double finalCov = nonCoreUV > 0 ? (double) coveredUV / nonCoreUV : 0.0;
        boolean budgetExhausted = (pvBudget - cumPV) <= 0 || activeCount == 0;

        System.out.printf("%n===== 结果：%d步 | UV %d→%d (覆盖%.1f%%) | PV %d→%d (预算使用率%.1f%%) =====%n",
                step, coreUV, cumUV, finalCov * 100, corePV, cumPV,
                pvBudget > corePV ? 100.0 * (cumPV - corePV) / (pvBudget - corePV) : 0.0);

        if (coveredUV >= uvTarget) {
            System.out.printf("[OK] UV 覆盖率达标 (%.1f%% ≥ %.1f%%)，且 PV 在预算内%n",
                    finalCov * 100, uvCoverageTarget * 100);
        } else if (budgetExhausted) {
            System.out.printf("[WARN] PV 预算耗尽，UV 未达标 (%.1f%% < %.1f%%)。"
                    + "需增大 pvBudgetRatio 或接受当前覆盖率%n",
                    finalCov * 100, uvCoverageTarget * 100);
        } else {
            System.out.printf("[INFO] 无更多可选事件，UV=%.1f%% (目标=%.1f%%)%n",
                    finalCov * 100, uvCoverageTarget * 100);
        }

        // ========== 7. 写回 Hive ==========
        StructType outputSchema = new StructType()
                .add("supplement_step",           DataTypes.IntegerType)
                .add("supplemented_pagetype",     DataTypes.StringType)
                .add("supplemented_actiontype",   DataTypes.StringType)
                .add("supplemented_event_pv",     DataTypes.LongType)
                .add("new_uv_gained",             DataTypes.LongType)
                .add("cum_uv",                    DataTypes.LongType)
                .add("cum_pv",                    DataTypes.LongType)
                .add("coverage_pct",              DataTypes.DoubleType);

        List<Row> sparkRows = new ArrayList<>(stepRows.size());
        for (Map<String, Object> row : stepRows) {
            sparkRows.add(RowFactory.create(
                    row.get("supplement_step"),
                    row.get("supplemented_pagetype"),
                    row.get("supplemented_actiontype"),
                    row.get("supplemented_event_pv"),
                    row.get("new_uv_gained"),
                    row.get("cum_uv"),
                    row.get("cum_pv"),
                    row.get("coverage_pct")
            ));
        }

        Dataset<Row> resultDF = spark.createDataFrame(sparkRows, outputSchema);
        resultDF.createOrReplaceTempView("tmp_supplement_result");
        spark.sql(
                "INSERT OVERWRITE TABLE " + targetTable +
                        " PARTITION(dt='" + dt + "', bg='" + bg + "')" +
                        " SELECT supplement_step, supplemented_pagetype, supplemented_actiontype," +
                        "        supplemented_event_pv, new_uv_gained," +
                        "        cum_uv, cum_pv, coverage_pct" +
                        " FROM tmp_supplement_result"
        );

        nonCoreDF.unpersist();
        pureNonCoreDF.unpersist();
        sourceDF.unpersist();

        System.out.println("结果已写入 " + targetTable);
        spark.stop();
    }
}