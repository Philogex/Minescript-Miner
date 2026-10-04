package dev.philogex.miner.runtime;

import dev.philogex.miner.bridge.NativeBridge;

import static dev.philogex.miner.bridge.NativeBridge.*;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.Executor;

/** Client-thread state machine. The planner receives immutable snapshots only. */
public final class MinerController {
    public enum Phase { INACTIVE, WAITING, SOLVING, AIMING, SETTLING, MINING }
    public record Snapshot(Scan scan, Map<Block, Integer> states, long worldId, long capturedNs, double angularStep) {
        public Snapshot { states = Map.copyOf(states); }
    }
    public record Plan(Snapshot snapshot, Optional<Target> target, List<AimPoint> path,
                       long scanNs, long aimNs) {
        public Plan { path = List.copyOf(path); }
    }
    public record AppliedSample(double plannedMs, long appliedNs, double latenessMs) {}
    public interface WorldAccess {
        Optional<Snapshot> capture(long now);
        boolean valid(Snapshot snapshot, Target target);
        boolean hitting(Target target);
        void orient(double yaw, double pitch);
        void attack(boolean pressed);
        void failed(Throwable error);
    }
    @FunctionalInterface public interface Planner { Plan calculate(Snapshot snapshot); }

    private final WorldAccess world;
    private final Planner planner;
    private final Executor worker;
    private final AppliedSample[] applied = new AppliedSample[8192];
    private int appliedCount;
    private boolean active;
    private long generation, pendingGeneration, nextNs, deadlineNs;
    private CompletableFuture<Plan> pending;
    private Plan plan;
    private int point;
    private Phase phase = Phase.INACTIVE;
    private static final long IDLE_NS = 250_000_000L, SETTLE_NS = 50_000_000L;

    public MinerController(WorldAccess world, Planner planner, Executor worker) {
        this.world = world; this.planner = planner; this.worker = worker;
    }
    public void setActive(boolean enabled) {
        active = enabled; generation++;
        plan = null; nextNs = 0; point = 0;
        world.attack(false);
        phase = active ? Phase.WAITING : Phase.INACTIVE;
        // Keep an in-flight job until it finishes, preventing an unbounded
        // queue when toggled repeatedly. Its generation invalidates the result.
    }
    public boolean active() { return active; }
    public Phase phase() { return phase; }
    public List<AppliedSample> appliedSamples() {
        var result = new java.util.ArrayList<AppliedSample>();
        for (int i = Math.max(0, appliedCount - applied.length); i < appliedCount; i++) result.add(applied[i % applied.length]);
        return List.copyOf(result);
    }
    public void frame(long now) {
        if (pending != null && pending.isDone()) {
            var finished = pending; pending = null;
            if (active && pendingGeneration == generation) {
                try { accept(finished.join(), now); }
                catch (CompletionException error) { world.failed(error.getCause()); setActive(false); }
            }
        }
        if (!active) return;
        if (phase == Phase.AIMING) {
            if (!world.valid(plan.snapshot, plan.target.orElseThrow())) { restart(now); return; }
            if (now < deadlineNs) return;
            var sample = plan.path.get(point);
            world.orient(sample.yaw(), sample.pitch());
            applied[appliedCount++ % applied.length] = new AppliedSample(sample.tMs(), now, (now - deadlineNs) / 1e6);
            point++;
            if (point == plan.path.size()) { phase = Phase.SETTLING; deadlineNs = now + SETTLE_NS; }
            else {
                // Preserve the old executor's relative waits, including frame
                // lateness. Absolute-time interpolation is a separate experiment.
                deadlineNs = now + Math.round(Math.max(0, plan.path.get(point).tMs() - sample.tMs()) * 1e6);
            }
        } else if (phase == Phase.SETTLING && now >= deadlineNs) {
            var target = plan.target.orElseThrow();
            if (!world.valid(plan.snapshot, target)) { restart(now); return; }
            var last = plan.path.getLast(); world.orient(last.yaw(), last.pitch());
            if (world.hitting(target)) { world.attack(true); phase = Phase.MINING; }
            else restart(now);
        }
    }
    public void tick(long now) {
        if (!active) return;
        if (phase == Phase.MINING) {
            var target = plan.target.orElseThrow();
            if (!world.valid(plan.snapshot, target) || !world.hitting(target)) restart(now);
        }
        if (phase != Phase.WAITING || pending != null || now < nextNs) return;
        var snapshot = world.capture(now);
        if (snapshot.isEmpty()) { nextNs = now + IDLE_NS; return; }
        pendingGeneration = generation;
        pending = CompletableFuture.supplyAsync(() -> planner.calculate(snapshot.get()), worker);
        phase = Phase.SOLVING;
    }
    private void accept(Plan result, long now) {
        if (result.target.isEmpty() || result.path.isEmpty()) { restart(now); nextNs = now + IDLE_NS; return; }
        if (!world.valid(result.snapshot, result.target.get())) { restart(now); return; }
        plan = result; point = 0; phase = Phase.AIMING;
        // Old Python execution applies the first sample immediately.
        deadlineNs = now;
    }
    private void restart(long now) {
        world.attack(false); plan = null; phase = Phase.WAITING; nextNs = now;
    }
}
