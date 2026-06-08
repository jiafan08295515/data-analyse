package com.demo;

import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.util.*;

import static org.apache.spark.sql.functions.*;

/**
 * 非核心埋点逐级补充算法 V9 —— 双硬约束 + 可行性报告。
 *
 * 相比 V8 改动：
 *   PV 预算和 UV 覆盖率目标均为硬约束。
 *   1. 在预算内贪心，计算预算内的最大 UV 覆盖率
 *   2. 可行 → 输出结果写入 Hive
 *   3. 不可行 → 保持预算不变，报告预算内最大覆盖率，不写 Hive
 *
 * 参数：dt bg corePV totalPV [pvBudgetRatio=1.5] [uvCoverageTarget=1.0] [minNewUV=100] [maxPVPerUV=100000]
 */
public class EventSupplementByMinPvEntry_V9 {

    public static void main(String[] args) {
        if (args.length < 4) {
            System.err.println("必须提供 dt bg corePV totalPV 参数");
            System.err.println("  dt:         日期分区");
            System.err.println("  bg:         业务组");
            System.err.println("  corePV:     核心埋点 PV（外部传入）");
            System.err.println("  totalPV:    总 PV（外部传入）");
            System.err.println("可选: pvBudgetRatio(默认1.5) uvCoverageTarget(默认1.0)");
            System.err.println("      minNewUV(默认100) maxPVPerUV(默认100000)");
            System.exit(1);
        }

        String dt = args[0];
        String bg = args[1];
        long corePV = Long.parseLong(args[2]);
        long totalPV = Long.parseLong(args[3]);
        double pvBudgetRatio = args.length > 4 ? Double.parseDouble(args[4]) : 1.5;
        double uvCoverageTarget = args.length > 5 ? Double.parseDouble(args[5]) : 1.0;
        long minNewUV = args.length > 6 ? Long.parseLong(args[6]) : 100L;
        long maxPVPerUV = args.length > 7 ? Long.parseLong(args[7]) : 100000L;

        SparkSession spark = SparkSession.builder()
                .appName("EventSupplementByMinPvEntry_V9_" + dt + "_" + bg)
                .enableHiveSupport()
                .config("hive.exec.dynamic.partition.mode", "nonstrict")
                .getOrCreate();

        spark.conf().set("spark.sql.shuffle.partitions", "800");
        spark.conf().set("spark.sql.adaptive.enabled", "true");
        spark.conf().set("spark.sql.adaptive.coalescePartitions.enabled", "true");

        String sourceTable = "hdp_teu_dpd_wx_flow.jf_event_impact_daily_v4";
        String targetTable = "hdp_teu_dpd_wx_flow.jf_event_supplement_by_pv_rank_v5";

        // ========== 1. 读取源表 ==========
        Dataset<Row> sourceDF = spark.table(sourceTable)
                .filter(col("dt").equalTo(dt).and(col("bg").equalTo(bg)))
                .repartition(800, col("imei"))
                .cache();
        sourceDF.count();

        // ========== 2. UV 指标计算（PV 由外部传入）==========
        Row metrics = sourceDF.agg(
                countDistinct("imei").as("total_uv"),
                countDistinct(when(col("is_core").equalTo(1), col("imei"))).as("core_uv")
        ).head();

        long totalUV = metrics.getLong(0);
        long coreUV  = metrics.getLong(1);
        long nonCoreUV = totalUV - coreUV;
        long pvBudget = (long) (corePV * pvBudgetRatio);
        long uvTarget = (long) (totalUV * uvCoverageTarget);

        System.out.println("========================================");
        System.out.printf("  总UV=%d  总PV=%d%n", totalUV, totalPV);
        System.out.printf("  核心UV=%d  核心PV=%d  非核心UV=%d%n", coreUV, corePV, nonCoreUV);
        System.out.printf("  硬约束1: 总PV ≤ %d (核心PV × %.2f)%n", pvBudget, pvBudgetRatio);
        System.out.printf("  硬约束2: 总UV覆盖率 ≥ %.1f%% (≥ %d人)%n", uvCoverageTarget * 100, uvTarget);
        System.out.println("========================================");

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

        System.out.printf("事件数=%d  用户-事件对=%d%n", eventMeta.size(), userPairs.size());

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

        // ---- 初始化计数器（含质量过滤）----
        long[] eventRemainingNewUV = new long[numEvents];
        int[] activeEventIds = new int[numEvents];
        int activeCount = 0;
        int filteredByMinUV = 0;
        int filteredByPVRatio = 0;

        for (int i = 0; i < numEvents; i++) {
            long uv = eventUsersSets[i].size();
            eventRemainingNewUV[i] = uv;
            if (uv == 0) continue;

            if (uv < minNewUV) {
                filteredByMinUV++;
                continue;
            }
            if (eventPVs[i] / uv > maxPVPerUV) {
                filteredByPVRatio++;
                continue;
            }
            activeEventIds[activeCount++] = i;
        }

        System.out.printf("事件过滤: minNewUV=%d 过滤%d个 | maxPVPerUV=%d 过滤%d个 | 有效事件=%d%n",
                minNewUV, filteredByMinUV, maxPVPerUV, filteredByPVRatio, activeCount);

        eventKeyToId = null;
        Set<String> covered = new HashSet<>((int) (nonCoreUserEstimate * 1.5));

        // ========== 6. 双硬约束贪心求解 ==========
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
        int stepAtBudgetExhausted = -1;
        int stepAtUVTarget = -1;

        System.out.printf("Step %4d | 预算内 | %-20s %-25s | PV=+%10d | UV=+%8d | cumUV=%10d | cumPV=%15d | 覆盖=%.1f%%%n",
                0, "[基线]", "", 0, 0, cumUV, cumPV, 0.0);

        while (activeCount > 0) {
            long remaining = pvBudget - cumPV;

            // 记录 UV 达标点（cumUV 已含 coreUV，uvTarget 基于 totalUV）
            if (stepAtUVTarget == -1 && cumUV >= uvTarget) {
                stepAtUVTarget = step;
            }

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

            if (bestId == -1) {
                stepAtBudgetExhausted = step;
                break;
            }

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

            double covPct = totalUV > 0 ? (double) cumUV / totalUV : 0.0;

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

            System.out.printf("Step %4d | 预算内 | %-20s %-25s | PV=+%10d | UV=+%8d | cumUV=%10d | cumPV=%15d | 覆盖=%.1f%%%n",
                    step, eventPagetypes[bestId], eventActiontypes[bestId],
                    bestPV, bestNewUV, cumUV, cumPV, covPct * 100);
        }

        // 检查截止预算耗尽时的 UV 是否达标
        if (stepAtBudgetExhausted == -1) stepAtBudgetExhausted = step;
        if (stepAtUVTarget == -1 && cumUV >= uvTarget) stepAtUVTarget = step;

        long cumUVAtBudget = coreUV;
        for (int i = 1; i <= stepAtBudgetExhausted && i < stepRows.size(); i++) {
            Map<String, Object> r = stepRows.get(i);
            cumUVAtBudget = (long) r.get("cum_uv");
        }
        double covAtBudget = totalUV > 0 ? (double) cumUVAtBudget / totalUV : 0.0;

        boolean constraintsSatisfied = stepAtUVTarget != -1
                && stepAtUVTarget <= stepAtBudgetExhausted;

        // ========== 7. 判定与输出 ==========
        if (constraintsSatisfied) {
            // ---- 可行：输出预算内且 UV 达标的结果 ----
            System.out.println();
            System.out.println("========================================");
            System.out.printf("  [可行] 两个硬约束同时满足%n");
            System.out.printf("  UV 覆盖率 %.1f%% 达标 (目标 %.1f%%)，在第 %d 步达成%n",
                    covAtBudget * 100, uvCoverageTarget * 100, stepAtUVTarget);
            System.out.printf("  最终 PV=%d，预算使用率 %.1f%%%n",
                    cumPV, pvBudget > corePV ? 100.0 * (cumPV - corePV) / (pvBudget - corePV) : 0.0);
            System.out.println("========================================");

            writeToHive(spark, stepRows, targetTable, dt, bg);

        } else {
            // ---- 不可行：保持预算不变，报告预算内最大覆盖率，仍然写入 Hive ----
            long uvGap = uvTarget - cumUVAtBudget;

            System.out.println();
            System.out.println("╔══════════════════════════════════════════════════════════╗");
            System.out.println("║              可 行 性 报 告 —— 无 可 行 解               ║");
            System.out.println("╠══════════════════════════════════════════════════════════╣");
            System.out.printf("║  硬约束1: PV ≤ %d (%.2f×核心PV)%n", pvBudget, pvBudgetRatio);
            System.out.printf("║  硬约束2: 总UV覆盖率 ≥ %.1f%% (≥ %d人)%n", uvCoverageTarget * 100, uvTarget);
            System.out.println("╠══════════════════════════════════════════════════════════╣");
            System.out.printf("║  诊断结果%n");
            System.out.printf("║    当前预算内最大覆盖率:  %.1f%% (%d/%d)%n",
                    covAtBudget * 100, cumUVAtBudget, totalUV);
            System.out.printf("║    UV 缺口 (无法在预算内弥补):  %.1f%% (%d人)%n",
                    (double) uvGap / totalUV * 100, uvGap);
            System.out.println("╠══════════════════════════════════════════════════════════╣");
            System.out.printf("║  建议方案 (二选一)%n");
            System.out.printf("║    A. 放宽 PV 约束 → 调大 pvBudgetRatio%n");
            System.out.printf("║    B. 降低 UV 目标 → 将 uvCoverageTarget 调至 ≤ %.1f%%%n",
                    covAtBudget * 100);
            System.out.println("╚══════════════════════════════════════════════════════════╝");
            System.out.println();

            writeToHive(spark, stepRows, targetTable, dt, bg);
        }

        nonCoreDF.unpersist();
        pureNonCoreDF.unpersist();
        sourceDF.unpersist();
        spark.stop();
    }

    private static void writeToHive(SparkSession spark, List<Map<String, Object>> stepRows,
                                     String targetTable, String dt, String bg) {
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
        System.out.println("结果已写入 " + targetTable);
    }
}