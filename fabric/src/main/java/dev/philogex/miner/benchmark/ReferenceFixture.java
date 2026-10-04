package dev.philogex.miner.benchmark;

import dev.philogex.miner.bridge.NativeBridge;
import dev.philogex.miner.config.AimConfig;

import static dev.philogex.miner.bridge.NativeBridge.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Properties;

/** Shared, language-neutral fixture format used by tests and benchmarks. */
public record ReferenceFixture(String name, Properties values, Scan scan) {
    public static ReferenceFixture read(Path file) throws IOException {
        var values = new Properties();
        try (var reader = Files.newBufferedReader(file)) { values.load(reader); }
        int side = Integer.parseInt(values.getProperty("side"));
        var pose = parseNumbers(values.getProperty("pose"));
        short[] shapes = new short[side * side * side];
        for (String pair : values.getProperty("shapes").split(",")) {
            if (pair.isEmpty()) continue;
            var parts = pair.split(":"); shapes[Integer.parseInt(parts[0])] = (short) Integer.parseInt(parts[1]);
        }
        String targetText = values.getProperty("targets");
        String[] targetParts = targetText.isEmpty() ? new String[0] : targetText.split(",");
        short[] targets = new short[targetParts.length];
        for (int i = 0; i < targets.length; i++) targets[i] = (short) Integer.parseInt(targetParts[i]);
        var scan = new Scan(new Vector(pose[0], pose[1], pose[2]), new Orientation(pose[3], pose[4]), side,
            Double.parseDouble(values.getProperty("reach")), shapes, targets);
        return new ReferenceFixture(file.getFileName().toString(), values, scan);
    }
    public double[] numbers(String key) { return parseNumbers(values.getProperty(key)); }
    private static double[] parseNumbers(String text) {
        if (text == null || text.isEmpty()) return new double[0];
        return Arrays.stream(text.split(",")).mapToDouble(Double::parseDouble).toArray();
    }
    public long seed() { return Long.parseUnsignedLong(values.getProperty("seed")); }
    public double step() { return Double.parseDouble(values.getProperty("step")); }
    public AimConfig config(String model) {
        return AimConfig.fromPayloads(model, numbers("minimum"), numbers("sigma"), numbers("feedback"));
    }
}
