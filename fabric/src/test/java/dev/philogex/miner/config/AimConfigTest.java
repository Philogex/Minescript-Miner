package dev.philogex.miner.config;

import static org.junit.jupiter.api.Assertions.*;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

class AimConfigTest {
    @Test void readsExistingConfiguration() throws Exception {
        var config = AimConfig.load(Path.of(System.getProperty("miner.project.dir"), "aim_config.txt"));
        assertEquals("geometry_feedback_sigmadrift", config.model());
        assertEquals(2, config.modelCode());
        assertArrayEquals(new double[]{80, 110, 60, 450, 120}, config.minimumPayload());
    }
    @Test void acceptsLegacyTopLevelAndComments() {
        var config = AimConfig.parse(List.of("fitts_a_ms: 123 # comment", "sigmadrift[", "target_width: 31", "]"));
        assertEquals(123, config.minimumPayload()[0]); assertEquals(31, config.sigmaPayload()[2]);
    }
    @Test void rejectsUnknownAndInvalidValues() {
        for (String text : List.of("unknown: 1", "aim_model: unknown", "sample_hz: 0", "sample_hz: 2.5", "fallback_angular_step_deg: NaN"))
            assertThrows(IllegalArgumentException.class, () -> AimConfig.parse(List.of(text)), text);
        assertThrows(IllegalArgumentException.class, () -> AimConfig.parse(List.of("geometry_feedback_sigmadrift[", "max_corrections: 65", "]")));
    }
    @Test void sensitivityMatchesExistingConvention() { assertEquals(.6144, AimConfig.angularStep(1), 1e-12); }
}
