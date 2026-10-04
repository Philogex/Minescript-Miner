package dev.philogex.miner.bridge;

import static org.junit.jupiter.api.Assertions.*;
import java.io.*;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

@Timeout(20)
class AnalysisServerTest {
    @Test void packagedJarServesDaqRequestsAndRecoversFromInvalidRequest() throws Exception {
        String executable = System.getProperty("os.name").startsWith("Windows") ? "java.exe" : "java";
        var process = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", executable).toString(),
            "--enable-native-access=ALL-UNNAMED", "-cp", System.getProperty("miner.jar.path"),
            "dev.philogex.miner.bridge.AnalysisServer").redirectError(ProcessBuilder.Redirect.INHERIT).start();
        try (var input = new DataInputStream(process.getInputStream());
             var output = new DataOutputStream(process.getOutputStream())) {
            var hello = request(input, output, Map.of("op", "hello", "protocol", 1, "model", "minimum_jerk"));
            assertEquals(39L, hello.get("max_cube_side"));
            assertEquals(64, ((String)map(hello.get("backend")).get("jar_sha256")).length());
            var config = map(hello.get("config"));
            assertEquals("minimum_jerk", config.get("aim_model"));
            assertEquals(120L, map(config.get("minimum_jerk")).get("sample_hz"));
            AnalysisProtocol.writeFrame(output, Map.of("op", "scan", "side", 41));
            assertTrue(map(AnalysisProtocol.readFrame(input)).containsKey("error"));
            var states = new ArrayList<>(Collections.nCopies(27, "minecraft:air"));
            states.set(16, "minecraft:diamond_ore");
            var target = request(input, output, Map.of("op", "scan", "pose", List.of(.5, .5, .5, 0., 0.),
                "side", 3, "reach", 4.8, "states", states, "targets", List.of(16)));
            assertEquals(List.of(0L, 0L, 1L), target.get("target_block"));
            assertEquals("north", target.get("face_id"));
            var aim = request(input, output, Map.of("op", "aim", "start", List.of(10., -5.),
                "metrics", target.get("metrics"), "components", target.get("visible_components"),
                "step", .15, "seed", "18446744073709551615", "model", "minimum_jerk"));
            assertEquals(2, ((List<?>)aim.get("points")).size());
            assertEquals(Map.of(), aim.get("diagnostics"));
        } finally {
            process.getOutputStream().close();
            if (!process.waitFor(2, TimeUnit.SECONDS)) { process.destroyForcibly(); process.waitFor(); }
        }
        assertEquals(0, process.exitValue(), "EOF should stop the headless process cleanly");
    }
    @Test void rejectsProtocolMismatchAndRequestsBeforeHello() {
        var server = new AnalysisServer();
        assertThrows(IllegalStateException.class, () -> server.handle(Map.of("op", "shape_ids", "states", List.of())));
        assertThrows(IllegalArgumentException.class, () -> server.handle(Map.of("op", "hello", "protocol", 2)));
    }
    @Test void configAndSensitivityValidationAreOwnedByJava() throws Exception {
        var server = new AnalysisServer();
        server.handle(Map.of("op", "hello", "protocol", 1, "model", "sigmadrift"));
        assertEquals(.15, (double)server.handle(Map.of("op", "angular_step", "sensitivity", .5)), 1e-12);
        for (double value : new double[]{-.1, 1.1, Double.NaN})
            assertThrows(IllegalArgumentException.class, () -> server.handle(Map.of("op", "angular_step", "sensitivity", value)));
        assertThrows(IllegalArgumentException.class, () -> server.handle(Map.of("op", "scan", "side", 3,
            "states", Collections.nCopies(27, "minecraft:air"), "targets", List.of(65536))));
    }
    @SuppressWarnings("unchecked") private static Map<String, Object> map(Object value) { return (Map<String, Object>)value; }
    private static Map<String, Object> request(DataInputStream input, DataOutputStream output, Map<String, Object> value) throws IOException {
        AnalysisProtocol.writeFrame(output, value);
        var response = map(AnalysisProtocol.readFrame(input));
        assertFalse(response.containsKey("error"), () -> "Analysis failed: " + response);
        return map(response.get("result"));
    }
}
