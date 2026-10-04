package dev.philogex.miner.bridge;

import static dev.philogex.miner.bridge.NativeBridge.*;
import dev.philogex.miner.bridge.NativeBridge.Vector;
import dev.philogex.miner.catalog.GeneratedCatalog;
import dev.philogex.miner.catalog.ShapeCatalog;
import dev.philogex.miner.config.AimConfig;
import java.io.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;

/** Headless DAQ adapter. Runs the same catalog/config/JNI API as the mod in one persistent JVM. */
public final class AnalysisServer {
    private AimConfig config;
    private static final String[] DIAGNOSTICS = {
        "motor_target_yaw", "motor_target_pitch", "applied_margin_steps", "anchor_component_index",
        "directional_width_steps", "s_enter_steps", "s_anchor_steps", "s_exit_steps", "primary_endpoint_steps",
        "first_feedback_observation_ms", "first_feedback_latency_ms", "first_feedback_application_ms",
        "first_predicted_terminal_x_steps", "first_predicted_terminal_y_steps", "first_prediction_sigma_major_steps",
        "first_prediction_sigma_minor_steps", "feedback_check_count", "unsafe_prediction_count", "correction_count",
        "first_visible_entry_ms", "first_safe_entry_ms", "visible_entry_count", "visible_exit_count",
        "safe_entry_count", "safe_exit_count", "final_visible", "final_safe"
    };

    public static void main(String[] args) throws IOException {
        var server = new AnalysisServer();
        var input = new DataInputStream(new BufferedInputStream(System.in));
        var output = new DataOutputStream(new BufferedOutputStream(System.out));
        Object frame;
        while ((frame = AnalysisProtocol.readFrame(input)) != null) {
            Object response;
            try { response = Collections.singletonMap("result", server.handle(map(frame))); }
            catch (RuntimeException | IOException error) {
                response = Map.of("error", error.getClass().getSimpleName() + ": " + error.getMessage());
            }
            AnalysisProtocol.writeFrame(output, response);
        }
    }

    Object handle(Map<String, Object> request) throws IOException {
        String op = (String) request.get("op");
        if ("hello".equals(op)) {
            if (integer(request.get("protocol")) != AnalysisProtocol.VERSION)
                throw new IllegalArgumentException("Unsupported analysis protocol");
            Object path = request.get("config");
            var loaded = path == null ? AimConfig.defaults() : AimConfig.load(Path.of((String)path));
            config = loaded.withModel((String)request.get("model"));
            Path artifact;
            // URI decoding also supports spaces in checkout paths.
            try { artifact = Path.of(AnalysisServer.class.getProtectionDomain().getCodeSource().getLocation().toURI()); }
            catch (java.net.URISyntaxException error) { throw new IOException(error); }
            String sha = "development";
            if (Files.isRegularFile(artifact)) {
                try { sha = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(artifact))); }
                catch (java.security.NoSuchAlgorithmException error) { throw new IllegalStateException(error); }
            }
            String version = AnalysisServer.class.getPackage().getImplementationVersion();
            return Map.of("config", config.metadata(), "backend", Map.of(
                "package", "minecraft-miner", "version", version == null ? "development" : version,
                "protocol", AnalysisProtocol.VERSION, "jar_sha256", sha,
                "catalog_sha256", GeneratedCatalog.SHAPE_CATALOG_SHA256,
                "java_version", System.getProperty("java.version"), "platform", System.getProperty("os.name"),
                "architecture", System.getProperty("os.arch")),
                "max_cube_side", GeneratedCatalog.MAX_CUBE_SIDE, "full_cube_shape_id", GeneratedCatalog.SHAPE_FULL_CUBE);
        }
        if (config == null) throw new IllegalStateException("hello required before analysis requests");
        return switch (op) {
            case "shape_ids" -> list(request.get("states")).stream().map(state -> ShapeCatalog.shapeId((String)state)).toList();
            case "angular_step" -> {
                double sensitivity = number(request.get("sensitivity"));
                if (!Double.isFinite(sensitivity) || sensitivity < 0 || sensitivity > 1)
                    throw new IllegalArgumentException("Sensitivity outside [0, 1]");
                yield AimConfig.angularStep(sensitivity);
            }
            case "scan" -> {
                int side = integer(request.get("side"));
                if (side < 1 || side > GeneratedCatalog.MAX_CUBE_SIDE) throw new IllegalArgumentException("Invalid cube side");
                var states = list(request.get("states"));
                if (states.size() != side * side * side) throw new IllegalArgumentException("Invalid block-state count");
                short[] shapes = new short[states.size()];
                for (int i = 0; i < shapes.length; i++) shapes[i] = (short)ShapeCatalog.shapeId((String)states.get(i));
                var indices = list(request.get("targets"));
                short[] targets = new short[indices.size()];
                for (int i = 0; i < targets.length; i++) {
                    int index = integer(indices.get(i));
                    if (index < 0 || index >= shapes.length || index > 65535) throw new IllegalArgumentException("Invalid target index");
                    targets[i] = (short)index;
                }
                var pose = list(request.get("pose"));
                if (pose.size() != 5) throw new IllegalArgumentException("Expected five pose coordinates");
                yield acquire(new Scan(vector(pose), new Orientation(number(pose.get(3)), number(pose.get(4))),
                    side, number(request.get("reach")), shapes, targets)).map(AnalysisServer::targetMetadata).orElse(null);
            }
            case "aim" -> {
                var start = list(request.get("start"));
                var metrics = list(request.get("metrics"));
                if (start.size() != 2 || metrics.size() != 6) throw new IllegalArgumentException("Invalid aim metrics");
                var components = list(request.get("components")).stream()
                    .map(component -> list(component).stream().map(v -> vector(list(v))).toList()).toList();
                var target = new Target(number(metrics.get(0)), number(metrics.get(1)), number(metrics.get(2)),
                    number(metrics.get(3)), number(metrics.get(4)), number(metrics.get(5)), null, null, null, components);
                var result = generate(new Orientation(number(start.get(0)), number(start.get(1))), target,
                    config.withModel((String)request.get("model")), number(request.get("step")),
                    Long.parseUnsignedLong((String)request.get("seed")));
                var diagnostics = new LinkedHashMap<String, Object>();
                if (!result.diagnostics().isEmpty()) {
                    if (result.diagnostics().size() != DIAGNOSTICS.length) throw new IllegalStateException("Invalid diagnostics count");
                    for (int i = 0; i < DIAGNOSTICS.length; i++) {
                        double value = result.diagnostics().get(i);
                        if (i >= 25) diagnostics.put(DIAGNOSTICS[i], value != 0);
                        else if (i == 3 || i >= 16 && i <= 18 || i >= 21) diagnostics.put(DIAGNOSTICS[i], (int)value);
                        else diagnostics.put(DIAGNOSTICS[i], value);
                    }
                }
                yield Map.of("points", result.points().stream().map(p -> List.of(p.yaw(), p.pitch(), p.tMs())).toList(),
                    "diagnostics", diagnostics);
            }
            default -> throw new IllegalArgumentException("Unknown analysis operation: " + op);
        };
    }

    private static Map<String, Object> targetMetadata(Target target) {
        return Map.of("metrics", Arrays.stream(target.metrics()).boxed().toList(),
            "target_block", List.of(target.block().x(), target.block().y(), target.block().z()),
            "face_id", target.face(), "hit_point", List.of(target.hit().x(), target.hit().y(), target.hit().z()),
            "visible_components", target.components().stream().map(c -> c.stream().map(v -> List.of(v.x(), v.y(), v.z())).toList()).toList());
    }
    private static Vector vector(List<?> v) {
        if (v.size() < 3) throw new IllegalArgumentException("Expected three vector coordinates");
        return new Vector(number(v.get(0)), number(v.get(1)), number(v.get(2)));
    }
    private static double number(Object value) { return ((Number)value).doubleValue(); }
    private static int integer(Object value) {
        double v = number(value);
        if (!Double.isFinite(v) || v != Math.rint(v) || v < Integer.MIN_VALUE || v > Integer.MAX_VALUE)
            throw new IllegalArgumentException("Expected int32 value");
        return (int)v;
    }
    @SuppressWarnings("unchecked") private static Map<String, Object> map(Object value) { return (Map<String, Object>)value; }
    private static List<?> list(Object value) { return (List<?>)value; }
}
