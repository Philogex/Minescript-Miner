package dev.philogex.miner.runtime;

import dev.philogex.miner.bridge.NativeBridge;

import static dev.philogex.miner.bridge.NativeBridge.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class MinerControllerTest {
    private static final Target TARGET = new Target(10, 0, 1, 1, 2, 1,
        new Block(0, 0, 1), "north", new Vector(.5, .5, 1), List.of());
    private static final MinerController.Snapshot SNAPSHOT = new MinerController.Snapshot(
        new Scan(new Vector(.5, .5, .5), new Orientation(0, 0), 3, 4.8, new short[27], new short[0]), Map.of(), 1, 0, .15);
    private static MinerController.Plan plan(MinerController.Snapshot snapshot) {
        return new MinerController.Plan(snapshot, Optional.of(TARGET),
            List.of(new AimPoint(0, 0, 0), new AimPoint(10, 0, 10)), 0, 0);
    }
    private static class World implements MinerController.WorldAccess {
        boolean valid = true, hitting = true, attack;
        int captures;
        Throwable failure;
        List<Double> orientations = new ArrayList<>();
        public Optional<MinerController.Snapshot> capture(long now) { captures++; return Optional.of(SNAPSHOT); }
        public boolean valid(MinerController.Snapshot snapshot, Target target) { return valid; }
        public boolean hitting(Target target) { return hitting; }
        public void orient(double yaw, double pitch) { orientations.add(yaw); }
        public void attack(boolean pressed) { attack = pressed; }
        public void failed(Throwable error) { failure = error; }
    }
    @Test void relativeTimingSettleAndReleaseAfterBlockChange() {
        var world = new World(); var controller = new MinerController(world, MinerControllerTest::plan, Runnable::run);
        controller.setActive(true); controller.tick(0); controller.frame(0);
        assertEquals(List.of(0.0), world.orientations);
        controller.frame(9_000_000); assertEquals(1, world.orientations.size());
        controller.frame(15_000_000); assertEquals(List.of(0.0, 10.0), world.orientations);
        assertEquals(5, controller.appliedSamples().getLast().latenessMs());
        controller.frame(64_000_000); assertFalse(world.attack);
        controller.frame(65_000_000); assertTrue(world.attack); assertEquals(3, world.orientations.size());
        world.valid = false; controller.tick(70_000_000); assertFalse(world.attack);
    }
    @Test void disablingDuringAimPreventsFurtherWrites() {
        var world = new World(); var controller = new MinerController(world, MinerControllerTest::plan, Runnable::run);
        controller.setActive(true); controller.tick(0); controller.frame(0);
        controller.setActive(false); controller.frame(100_000_000); controller.tick(100_000_000);
        assertEquals(List.of(0.0), world.orientations); assertFalse(world.attack);
    }
    @Test void discardedJobsCannotBuildAnUnboundedQueue() {
        var world = new World(); var jobs = new ArrayList<Runnable>();
        var controller = new MinerController(world, MinerControllerTest::plan, jobs::add);
        controller.setActive(true); controller.tick(0);
        controller.setActive(false); controller.setActive(true); controller.tick(0);
        assertEquals(1, jobs.size());
        jobs.removeFirst().run(); controller.frame(0);
        assertTrue(world.orientations.isEmpty());
        controller.tick(0); assertEquals(1, jobs.size());
        jobs.removeFirst().run(); controller.frame(0); assertEquals(List.of(0.0), world.orientations);
    }
    @Test void invalidSnapshotAndFailedRayCheckDoNotAttack() {
        var world = new World(); world.valid = false;
        var controller = new MinerController(world, MinerControllerTest::plan, Runnable::run);
        controller.setActive(true); controller.tick(0); controller.frame(0);
        assertTrue(world.orientations.isEmpty()); assertFalse(world.attack);
        world.valid = true; world.hitting = false;
        controller.tick(0); controller.frame(0); controller.frame(10_000_000); controller.frame(60_000_000);
        assertFalse(world.attack); assertEquals(MinerController.Phase.WAITING, controller.phase());
    }
    @Test void workerFailuresStopTheMinerAndReleaseAttack() {
        var world = new World();
        var controller = new MinerController(world, snapshot -> { throw new IllegalStateException("failure"); }, Runnable::run);
        controller.setActive(true); controller.tick(0); controller.frame(0);
        assertFalse(controller.active()); assertFalse(world.attack); assertEquals("failure", world.failure.getMessage());
    }
    @Test void emptyScanUsesIdleDelay() {
        var world = new World();
        var controller = new MinerController(world,
            snapshot -> new MinerController.Plan(snapshot, Optional.empty(), List.of(), 0, 0), Runnable::run);
        controller.setActive(true); controller.tick(0); controller.frame(0);
        controller.tick(249_000_000); assertEquals(1, world.captures);
        controller.tick(250_000_000); assertEquals(2, world.captures);
    }
    @Test void noSnapshotSkipsPlannerAndUsesIdleDelay() {
        var world = new World() {
            public Optional<MinerController.Snapshot> capture(long now) { captures++; return Optional.empty(); }
        };
        var controller = new MinerController(world, snapshot -> { fail("Planner called without a snapshot"); return plan(snapshot); }, Runnable::run);
        controller.setActive(true); controller.tick(0); controller.frame(0);
        controller.tick(249_000_000);
        assertEquals(1, world.captures); assertTrue(world.orientations.isEmpty()); assertFalse(world.attack);
        controller.tick(250_000_000); assertEquals(2, world.captures);
    }
    @Test void disablingDuringSettlePreventsMining() {
        var world = new World(); var controller = new MinerController(world, MinerControllerTest::plan, Runnable::run);
        controller.setActive(true); controller.tick(0); controller.frame(0); controller.frame(10_000_000);
        assertEquals(MinerController.Phase.SETTLING, controller.phase());
        controller.setActive(false); controller.frame(60_000_000);
        assertEquals(List.of(0.0, 10.0), world.orientations); assertFalse(world.attack);
    }
    @Test void disablingDuringMiningReleasesAttack() {
        var world = new World(); var controller = new MinerController(world, MinerControllerTest::plan, Runnable::run);
        controller.setActive(true); controller.tick(0); controller.frame(0); controller.frame(10_000_000); controller.frame(60_000_000);
        assertTrue(world.attack);
        controller.setActive(false); controller.tick(70_000_000); controller.frame(70_000_000);
        assertFalse(world.attack); assertEquals(MinerController.Phase.INACTIVE, controller.phase());
    }
    @Test void invalidatingSnapshotDuringAimStopsFurtherCameraWrites() {
        var world = new World(); var controller = new MinerController(world, MinerControllerTest::plan, Runnable::run);
        controller.setActive(true); controller.tick(0); controller.frame(0);
        world.valid = false; controller.frame(10_000_000);
        assertEquals(List.of(0.0), world.orientations); assertFalse(world.attack);
        assertEquals(MinerController.Phase.WAITING, controller.phase());
    }
}
