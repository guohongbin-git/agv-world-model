"""B4 工业强基线 (B-Industrial) 定向用例 — GPT 尽调攻击一 (B0 弱基线) 补强。

B4 = 工业标准异常处理: 设备字 ACK (有限重询) + op_id 幂等去重 +
journal 前置日志 + 重启对账 (超时→failed+回滚) + 失败腿重试 ≤2。
红线: 完成判据不读物理真值/传感器 (与 B3 证据判据严格分道);
twin/services 生产代码零改动。
"""
import experiments.fault_ablation as fa


def _make_ctx(fault="S01", seed=0):
    ctx = fa.Ctx("B4", fault, seed)
    ctx.ensure_world()
    return ctx


def _pick_key(ctx):
    return f"{ctx.task_ref}:pick:PAL-001:A-L3-C1"


def test_b4_ack_idempotent_dedup():
    """ACK 确认 + op_id 幂等: 重复回报去重, 零重复事件/记账。"""
    ctx = _make_ctx()
    try:
        out1 = fa.commit_attempt(ctx, "PICK")
        assert out1["status"] == "committed"
        key = _pick_key(ctx)
        assert ctx.plc.ack(key), "设备状态字应已置位 (ACK)"
        assert ctx.b4_ops[key] == "COMMITTED"
        n_events = fa._pallet_moved_count(ctx)
        out2 = fa.commit_attempt(ctx, "PICK")          # 同 op 重复回报
        assert out2["status"] == "committed"
        assert "去重" in out2["reason"]
        assert fa._pallet_moved_count(ctx) == n_events  # 零重复事件
        assert fa.ledger_pallet_loc(ctx.world) == "ROBOT:F01"
    finally:
        ctx.cleanup()


def test_b4_journal_restart_recovery():
    """journal 前置日志: ACK 后提交前崩溃 → 重启依 journal 幂等重放补账。"""
    ctx = _make_ctx("D03")
    try:
        out = fa.commit_attempt(ctx, "PICK", crash=("pre_commit",),
                                injection_point="before_begin")
        assert out["status"] == "crashed"
        assert fa.ledger_pallet_loc(ctx.world) == "SLOT:A-L3-C1"  # 零残留
        ctx.cell.advance(2.0)
        ctx.checkpoint("crash")
        t_restart, rep = ctx.restart_b4()
        assert rep and rep[0]["decision"] == "committed_by_replay"
        assert rep[0]["recovered"]
        key = _pick_key(ctx)
        assert ctx.world.transfer_recorded(key)
        assert fa.ledger_pallet_loc(ctx.world) == "ROBOT:F01"     # 补账
        assert ctx.b4_ops[key] == "COMMITTED"
    finally:
        ctx.cleanup()


def test_b4_retry_cap_chain_failed():
    """重试上限: 设备字不置位 → 恰 1+2 次下发后 chain_failed, 账本零写入。"""
    ctx = _make_ctx("D02")
    try:
        out = fa.commit_attempt(ctx, "PICK", actuate=False)
        assert out["status"] == "chain_failed"
        key = _pick_key(ctx)
        issued = [e for e in ctx.journal.entries
                  if e["kind"] == "ISSUED" and e["op"] == key]
        assert len(issued) == fa.B4_MAX_RETRIES + 1
        assert ctx.b4_ops[key] == "FAILED"
        assert fa.ledger_pallet_loc(ctx.world) == "SLOT:A-L3-C1"
        assert not ctx.world.transfer_recorded(key)
        assert ctx.safe_failure
    finally:
        ctx.cleanup()


def test_b4_reconcile_timeout_rollback():
    """重启对账超时策略: 下发后执行前崩溃 → 对账标 failed+回滚 (不挂死)。"""
    ctx = _make_ctx("D02")
    try:
        out = fa.commit_attempt(ctx, "PICK", crash=("pre_exec",),
                                injection_point="after_issue")
        assert out["status"] == "crashed"
        ctx.cell.advance(2.0)
        t_restart, rep = ctx.restart_b4()
        assert rep and rep[0]["decision"] == "timeout_failed_rolled_back"
        assert rep[0]["recovered"]
        key = _pick_key(ctx)
        assert ctx.b4_ops[key] == "FAILED"
        assert fa.ledger_pallet_loc(ctx.world) == "SLOT:A-L3-C1"
        assert not ctx.world.transfer_recorded(key)
    finally:
        ctx.cleanup()


def test_b4_journal_file_reload():
    """journal 落盘可重放: 新实例从文件恢复全部条目。"""
    ctx = _make_ctx()
    try:
        ctx.journal.append("ISSUED", t=1.0, op="op-x", spec={"k": 1})
        ctx.journal.append("COMMITTED", t=2.0, op="op-x", attempt=1)
        j2 = fa._Journal(ctx.journal.path)
        assert [e["kind"] for e in j2.entries] == ["ISSUED", "COMMITTED"]
        assert j2.entries[0]["spec"] == {"k": 1}
    finally:
        ctx.cleanup()


def test_b4_s01_strong_baseline_vs_b0():
    """S01 单跑对照: B0 虚报完成即 unsafe 提交; B4 依协议执行后安全闭合。"""
    r0 = fa.run_one("B0", "S01", 0)
    r4 = fa.run_one("B4", "S01", 0)
    assert set(r0["metrics"]) == set(r4["metrics"])        # 输出同 schema
    assert r0["metrics"]["unsafe"] >= 1
    assert r4["metrics"]["unsafe"] == 0
    assert r4["metrics"]["task_completion"] == 1
    assert r4["metrics"]["divergence_duration_s"] == 0


def test_b4_all_scenarios_same_schema():
    """B4 全场景 (10) 同管线可跑, 指标 schema 与 B0 完全一致。"""
    base = fa.run_one("B0", "S01", 0)["metrics"]
    for s in fa.SCRIPTS:
        r = fa.run_one("B4", s, 0)
        assert set(r["metrics"]) == set(base), s
        assert isinstance(r["metrics"]["commits"], int)
