package com.demo;

import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

import java.util.*;
import java.util.BitSet;

import static org.apache.spark.sql.functions.*;

/**
 * 非核心埋点逐级补充算法 V9 —— 双硬约束 + 可行性报告。
 *
 * 相比 V8 改动：
 *   - PV 预算和 UV 覆盖率目标均为硬约束。
 *   - corePV 从源表 jf_event_impact_daily_v4 的 core_pv 字段直接读取（不再外部传入）。
 *   - 终止条件：pvBudgetRatio 超过 5 倍 或 uvCoverageTarget 达到 0.995。
 *   1. 在预算内贪心，计算预算内的最大 UV 覆盖率
 *   2. 可行 → 输出结果写入 Hive
 *   3. 不可行 → 仍写入 Hive，同时输出可行性报告
 *
 * 参数：dt bg [pvBudgetRatio=5.0] [uvCoverageTarget=0.995] [minNewUV=10] [maxPVPerUV=20000] [maxEventPV=100000000]
 */
public class EventSupplementByMinPvEntry_V9 {

    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("必须提供 dt bg 参数");
            System.err.println("  dt:         日期分区");
            System.err.println("  bg:         业务组");
            System.err.println("可选: pvBudgetRatio(默认5.0) uvCoverageTarget(默认0.995)");
            System.err.println("      minNewUV(默认10) maxPVPerUV(默认20000) maxEventPV(默认10000000=1千万)");
            System.exit(1);
        }

        String dt = args[0];
        String bg = args[1];
        double pvBudgetRatio = args.length > 2 ? Double.parseDouble(args[2]) : 5.0;
        double uvCoverageTarget = args.length > 3 ? Double.parseDouble(args[3]) : 0.995;
        long minNewUV = args.length > 4 ? Long.parseLong(args[4]) : 10L;
        long maxPVPerUV = args.length > 5 ? Long.parseLong(args[5]) : 20000L;
        long maxEventPV = args.length > 6 ? Long.parseLong(args[6]) : 10000000L;

        SparkSession spark = SparkSession.builder()
                .appName("EventSupplementByMinPvEntry_V9_" + dt + "_" + bg)
                .enableHiveSupport()
                .config("hive.exec.dynamic.partition.mode", "nonstrict")
                .getOrCreate();

        spark.conf().set("spark.sql.shuffle.partitions", "800");
        spark.conf().set("spark.sql.adaptive.enabled", "true");
        spark.conf().set("spark.sql.adaptive.coalescePartitions.enabled", "true");

        String sourceTable = "hdp_teu_dpd_wx_flow.dws_wmda_event_impact_daily_data";
//        String targetTable = "hdp_teu_dpd_wx_flow.jf_event_supplement_by_pv_rank_v5";
        String targetTable = "hdp_teu_dpd_wx_flow.dim_wmda_event_supplement_data";

        // ========== 1. 读取源表 ==========
        Dataset<Row> sourceDF = spark.table(sourceTable)
                .filter(col("dt").equalTo(dt).and(col("bg").equalTo(bg)))
                .repartition(800, col("imei"))
                .cache();
        sourceDF.count();

        // ========== 2. UV 指标计算（corePV 从表中 core_pv 字段读取）==========
        Row metrics = sourceDF.agg(
                countDistinct("imei").as("total_uv"),
                countDistinct(when(col("is_core").equalTo(1), col("imei"))).as("core_uv"),
                coalesce(max("core_pv"), lit(0L)).as("core_pv")
        ).head();

        long totalUV = metrics.getLong(0);
        long coreUV  = metrics.getLong(1);
        long corePV  = metrics.getLong(2);
        if (corePV == 0) {
            System.err.println("核心PV=0，无法计算预算，请检查源表 core_pv 字段");
            System.exit(1);
        }
        long nonCoreUV = totalUV - coreUV;
        long pvBudget = (long) (corePV * pvBudgetRatio);
        long uvTarget = (long) (totalUV * uvCoverageTarget);

        System.out.println("========================================");
        System.out.printf("  总UV=%d%n", totalUV);
        System.out.printf("  核心UV=%d  核心PV=%d  非核心UV=%d%n", coreUV, corePV, nonCoreUV);
        System.out.printf("  硬约束1: 总PV ≤ %d (核心PV × %.2f)%n", pvBudget, pvBudgetRatio);
        System.out.printf("  硬约束2: 总UV覆盖率 ≥ %.1f%% (≥ %d人)%n", uvCoverageTarget * 100, uvTarget);
        System.out.println("========================================");

        // ========== 3. 核心用户列表 ==========
        Dataset<Row> coreUserDF = sourceDF
                .filter(col("is_core").equalTo(1))
                .select("imei").distinct();

        // ========== 4. 非核心用户数据 ==========
        // nonCoreDF 仅被 pureNonCoreDF 消费一次, 无需缓存
        // pv 列在下游链路中从未使用, 不 select 以减少数据量
        Dataset<Row> nonCoreDF = sourceDF
                .filter(col("is_core").equalTo(0))
                .select(col("pagetype"), col("actiontype"), col("imei"));

        Dataset<Row> pureNonCoreDF = nonCoreDF
                .join(broadcast(coreUserDF), nonCoreDF.col("imei").equalTo(coreUserDF.col("imei")), "left_anti")
                .cache();
        pureNonCoreDF.count(); // 物化缓存

        // ========== 5. 事件元数据 + 两阶段聚合消除热点倾斜 ==========
        // 阶段1: 加盐部分聚合 — 热点 key 被盐值打散到多个 reducer
        // 阶段2: 去盐最终聚合 — 部分结果合并
        // 优化: 聚合前裁剪列, 仅保留 pagetype/actiontype/pv, 减少 Shuffle 数据量
        // Row: [0:pagetype, 1:actiontype, 2:event_pv(Long), 3:event_count(Long)]
        final int SALT_MOD = 800;
        List<Row> eventMeta = sourceDF
                .select(col("pagetype"), col("actiontype"), col("pv"))
                .withColumn("salt", pmod(hash(col("pagetype"), col("actiontype")), lit(SALT_MOD)))
                .groupBy(col("pagetype"), col("actiontype"), col("salt"))
                .agg(sum("pv").as("partial_pv"),
                     count("*").as("partial_count"))
                .groupBy(col("pagetype"), col("actiontype"))
                .agg(sum("partial_pv").as("event_pv"),
                     sum("partial_count").as("event_count"))
                .collectAsList();

        int numEvents = eventMeta.size();
        Map<String, Integer> eventKeyToId = new HashMap<>(numEvents * 2);
        String[] eventPagetypes = new String[numEvents];
        String[] eventActiontypes = new String[numEvents];
        long[] eventPVs = new long[numEvents];

        int approxFiltered = 0;
        java.util.LinkedHashSet<String> goodEventKeySet = new java.util.LinkedHashSet<>();

        // 宽松阈值: count ≥ UV 总是成立, 所以预过滤是保守的 (不会错杀)
        long looseMinNewUV = Math.max(1, minNewUV / 5);
        long looseMaxPVPerUV = maxPVPerUV * 5;

        for (int i = 0; i < numEvents; i++) {
            Row r = eventMeta.get(i);
            eventKeyToId.put(r.getString(0) + "|" + r.getString(1), i);
            eventPagetypes[i] = r.getString(0);
            eventActiontypes[i] = r.getString(1);
            eventPVs[i] = r.getLong(2);
            long cnt = r.getLong(3); // 事件行数(≥ UV), 用于预过滤

            // ---- 宽松预过滤: 只筛掉明显无用的, 节省后续 collect 量 ----
            if (cnt == 0) continue;
            if (cnt < looseMinNewUV) { approxFiltered++; continue; }
            if (cnt > 0 && eventPVs[i] / cnt > looseMaxPVPerUV) { approxFiltered++; continue; }
            if (eventPVs[i] > maxEventPV) { approxFiltered++; continue; } // 全量PV超上限
            goodEventKeySet.add(r.getString(0) + "|" + r.getString(1));
        }
        eventMeta = null;

        int goodEventCount = goodEventKeySet.size();
        System.out.printf("事件数=%d | 宽松预过滤掉%d个 | 候选事件=%d (最终exact检查在后)%n",
                numEvents, approxFiltered, goodEventCount);

        // ---- 构建用户-事件关联（只收集通过质量检查的事件数据）----
        if (goodEventKeySet.isEmpty()) {
            System.err.println("所有事件均未通过质量过滤, 退出");
            System.exit(1);
        }

        // ---- 用 broadcast join 过滤 pureNonCoreDF, 只保留有效事件 ----
        java.util.List<Row> goodEventRows = new java.util.ArrayList<>(goodEventKeySet.size());
        java.util.Iterator<String> keyIter = goodEventKeySet.iterator();
        while (keyIter.hasNext()) {
            String[] parts = keyIter.next().split("\\|", 2);
            goodEventRows.add(RowFactory.create(parts[0], parts[1]));
        }
        goodEventKeySet.clear();

        StructType goodEventSchema = new StructType()
                .add("pagetype", DataTypes.StringType)
                .add("actiontype", DataTypes.StringType);
        Dataset<Row> goodEventsDF = spark.createDataFrame(goodEventRows, goodEventSchema);
        goodEventRows = null;

        Dataset<Row> filteredUserPairs = pureNonCoreDF
                .join(broadcast(goodEventsDF),
                      pureNonCoreDF.col("pagetype").equalTo(goodEventsDF.col("pagetype"))
                          .and(pureNonCoreDF.col("actiontype").equalTo(goodEventsDF.col("actiontype"))),
                      "inner")
                .select(pureNonCoreDF.col("pagetype"), pureNonCoreDF.col("actiontype"),
                        pureNonCoreDF.col("imei"))
                .distinct()
                .coalesce(200)
                .cache();
        long userPairCount = filteredUserPairs.count();
        System.out.printf("有效事件关联的用户-事件对数=%d%n", userPairCount);

        // ---- 用 Spark 并行聚合构建数据结构, 替代串行 toLocalIterator ----
        // Step A: 并行收集不重复 imei, 构建 String→int 映射
        List<Row> imeiRows = filteredUserPairs.select("imei").distinct().collectAsList();
        int nonCoreUserCount = imeiRows.size();
        Map<String, Integer> imeiToIntId = new HashMap<>(nonCoreUserCount * 2);
        int id = 0;
        for (Row r : imeiRows) {
            imeiToIntId.put(r.getString(0), id++);
        }
        imeiRows = null;

        // Step B: 加盐 groupBy + Driver 端合并, 避免热点事件 collect_list OOM
        // 阶段1: 按 (pagetype, actiontype, salt) 分组, 热点事件的用户被盐值打散
        final int EVENT_SALT_MOD = 100;
        List<Row> saltedEventUsers = filteredUserPairs
                .withColumn("event_salt", pmod(hash(col("pagetype"), col("actiontype")), lit(EVENT_SALT_MOD)))
                .groupBy("pagetype", "actiontype", "event_salt")
                .agg(collect_list("imei").as("imeis"))
                .collectAsList();

        int[][] eventUsersArrays = new int[numEvents][];
        for (int i = 0; i < numEvents; i++) {
            eventUsersArrays[i] = new int[0];
        }

        // 阶段2: Driver 端按 event 合并各盐值的 partial 列表, 同时构建倒排索引
        // 使用临时 Map<eventId, List<String>> 收集每个事件在各盐值下的 partial imei 列表
        Map<Integer, java.util.ArrayList<String>> eventImeisMerge = new HashMap<>();
        for (Row r : saltedEventUsers) {
            Integer eventId = eventKeyToId.get(r.getString(0) + "|" + r.getString(1));
            if (eventId == null) continue;

            List<String> partialImeis = r.getList(3); // [0:pagetype, 1:actiontype, 2:salt, 3:imeis]
            java.util.ArrayList<String> merged = eventImeisMerge.get(eventId);
            if (merged == null) {
                merged = new java.util.ArrayList<>(partialImeis.size());
                eventImeisMerge.put(eventId, merged);
            }
            merged.addAll(partialImeis);
        }
        saltedEventUsers = null;

        // 将合并结果转为 int[][] 并构建 imei→eventIds 倒排索引
        Map<Integer, List<Integer>> imeiToEventIds = new HashMap<>();
        for (Map.Entry<Integer, java.util.ArrayList<String>> entry : eventImeisMerge.entrySet()) {
            int eventId = entry.getKey();
            java.util.ArrayList<String> imeis = entry.getValue();
            int sz = imeis.size();
            int[] arr = new int[sz];
            int j = 0;
            for (String imei : imeis) {
                Integer imeiId = imeiToIntId.get(imei);
                arr[j++] = imeiId;

                List<Integer> list = imeiToEventIds.get(imeiId);
                if (list == null) {
                    list = new ArrayList<>(2);
                    imeiToEventIds.put(imeiId, list);
                }
                list.add(eventId);
            }
            eventUsersArrays[eventId] = arr;
        }
        eventImeisMerge = null;

        filteredUserPairs.unpersist();
        imeiToIntId = null; // 释放 String→int 映射

        // ---- 初始化计数器（exact UV 从 int[] 获取）----
        long[] eventRemainingNewUV = new long[numEvents];
        int[] activeEventIds = new int[numEvents];
        int activeCount = 0;
        int finalFilteredByMinUV = 0;
        int finalFilteredByPVRatio = 0;

        for (int i = 0; i < numEvents; i++) {
            long uv = eventUsersArrays[i].length; // exact 非核心UV, 和原V9一致
            eventRemainingNewUV[i] = uv;
            if (uv == 0) continue;

            // ---- exact 质量检查: 和原V9逻辑完全一致 ----
            if (uv < minNewUV) { finalFilteredByMinUV++; continue; }
            if (eventPVs[i] / uv > maxPVPerUV) { finalFilteredByPVRatio++; continue; }
            activeEventIds[activeCount++] = i;
        }

        System.out.printf("exact质量过滤: minNewUV=%d 过滤%d个 | maxPVPerUV=%d 过滤%d个 | 最终活跃=%d%n",
                minNewUV, finalFilteredByMinUV, maxPVPerUV, finalFilteredByPVRatio, activeCount);

        eventKeyToId = null;
        BitSet covered = new BitSet(nonCoreUserCount); // 优化①: BitSet 替代 HashSet<String>

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
            // ---- 优化③: UV 覆盖率达标时提前终止 ----
            if (cumUV >= uvTarget) {
                if (stepAtUVTarget == -1) stepAtUVTarget = step;
                System.out.printf("UV 覆盖率已达标 (≥%.1f%%)，提前终止贪心迭代%n", uvCoverageTarget * 100);
                break;
            }

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

            for (int uid : eventUsersArrays[bestId]) {
                if (covered.get(uid)) continue;
                covered.set(uid);
                List<Integer> affected = imeiToEventIds.get(uid);
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