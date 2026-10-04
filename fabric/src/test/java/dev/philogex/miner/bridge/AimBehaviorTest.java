package dev.philogex.miner.bridge;

import static dev.philogex.miner.bridge.NativeBridge.*;
import static org.junit.jupiter.api.Assertions.*;
import dev.philogex.miner.config.AimConfig;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Behavioral checks retained from the retired Python aim suite. */
class AimBehaviorTest {
    private static Target target(double effectiveWidth, List<List<Vector>> components) {
        return new Target(0, 0, 2, 2, 4, effectiveWidth, null, null, null, components);
    }
    private static List<List<Vector>> region() {
        return List.of(List.of(new Vector(-.25, -.25, 1), new Vector(.25, -.25, 1),
            new Vector(.25, .25, 1), new Vector(-.25, .25, 1)));
    }
    @Test void minimumJerkUsesEffectiveWidthInFittsDuration() {
        var config = AimConfig.parse(List.of("fitts_a_ms: 0", "fitts_b_ms: 100",
            "min_duration_ms: 0", "max_duration_ms: 1000"));
        var start = new Orientation(10, 0);
        var local = NativeBridge.generate(start, target(0, List.of()), config, .15, 123);
        var full = NativeBridge.generate(start, target(4, List.of()), config, .15, 123);
        assertTrue(full.points().getLast().tMs() < local.points().getLast().tMs());
        assertEquals(new AimPoint(0, 0, full.points().getLast().tMs()), full.points().getLast());
        assertTrue(full.diagnostics().isEmpty());
    }
    @Test void sigmaDriftPreservesEndpointsAndSeed() {
        var config = AimConfig.defaults().withModel("sigmadrift");
        var start = new Orientation(12, -4);
        var result = NativeBridge.generate(start, target(2, List.of()), config, .15, -1);
        assertEquals(result, NativeBridge.generate(start, target(2, List.of()), config, .15, -1));
        assertTrue(result.points().size() > 2);
        assertEquals(new AimPoint(12, -4, 0), result.points().getFirst());
        assertEquals(0, result.points().getLast().yaw());
        assertEquals(0, result.points().getLast().pitch());
        assertTrue(result.diagnostics().isEmpty());
    }
    @Test void feedbackRequiresVisibleComponents() {
        assertThrows(IllegalArgumentException.class, () -> NativeBridge.generate(new Orientation(10, -2),
            target(2, List.of()), AimConfig.defaults().withModel("geometry_feedback_sigmadrift"), .15, 12345));
    }
    @Test void feedbackDiagnosticsDescribeTheGeneratedPath() {
        var result = NativeBridge.generate(new Orientation(10, -2), target(2, region()),
            AimConfig.defaults().withModel("geometry_feedback_sigmadrift"), .15, 12345);
        var d = result.diagnostics();
        assertEquals(27, d.size());
        assertTrue(d.get(16) >= 1 && d.get(18) >= 0 && d.get(18) <= d.get(16));
        assertTrue(d.get(2) >= 0 && d.get(4) > 0);
        assertTrue(d.get(5) < d.get(6) && d.get(6) < d.get(7));
        assertEquals(d.get(9) + d.get(10), d.get(11), 1e-12);
        assertTrue(d.get(14) >= d.get(15));
        assertEquals(1, d.get(25));
        for (int i = 1; i < result.points().size(); i++)
            assertTrue(result.points().get(i).tMs() > result.points().get(i - 1).tMs());
    }
    @Test void feedbackScalesEndpointErrorWithDirectionalWidth() {
        var config = AimConfig.parse(List.of("aim_model: geometry_feedback_sigmadrift",
            "sigmadrift[", "overshoot_prob: 0", "]", "geometry_feedback_sigmadrift[",
            "feedback_latency_mean_ms: 75", "feedback_latency_stddev_ms: 0",
            "feedback_latency_min_ms: 75", "feedback_latency_max_ms: 75",
            "undershoot_width_min: 0.2", "undershoot_width_max: 0.2", "]"));
        var d = NativeBridge.generate(new Orientation(10, -2), target(2, region()), config, .15, 12345).diagnostics();
        assertEquals(.2, (d.get(6) - d.get(8)) / d.get(4), 1e-12);
        assertEquals(75, d.get(10));
        assertTrue(d.get(9) > 0);
    }
}
