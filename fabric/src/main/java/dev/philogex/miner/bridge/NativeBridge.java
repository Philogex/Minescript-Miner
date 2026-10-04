package dev.philogex.miner.bridge;

import dev.philogex.miner.catalog.GeneratedCatalog;
import dev.philogex.miner.config.AimConfig;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public final class NativeBridge {
    static {
        NativeLibrary.load();
        if (abiVersion0() != 1) throw new IllegalStateException("Unsupported miner JNI ABI");
        if (!GeneratedCatalog.SHAPE_CATALOG_SHA256.equals(catalogFingerprint0()))
            throw new IllegalStateException("Java/native shape catalogs differ");
    }
    private NativeBridge() {}

    public record Orientation(double yaw, double pitch) {}
    public record Vector(double x, double y, double z) {}
    public record Block(int x, int y, int z) {}
    public record AimPoint(double yaw, double pitch, double tMs) {}
    public record Target(double yaw, double pitch, double widthYaw, double widthPitch,
                         double distance, double effectiveWidth, Block block, String face,
                         Vector hit, List<List<Vector>> components) {
        public Target { components = components.stream().map(List::copyOf).toList(); }
        public double[] metrics() { return new double[]{yaw, pitch, widthYaw, widthPitch, distance, effectiveWidth}; }
    }
    public record AimResult(List<AimPoint> points, List<Double> diagnostics) {
        public AimResult { points = List.copyOf(points); diagnostics = List.copyOf(diagnostics); }
    }
    public record Scan(Vector eye, Orientation orientation, int side, double reach,
                       short[] shapes, short[] targets) {
        public Scan {
            if (side < 1 || side > GeneratedCatalog.MAX_CUBE_SIDE) throw new IllegalArgumentException("Invalid cube side");
            if (shapes.length != side * side * side) throw new IllegalArgumentException("Invalid cube payload");
            shapes = shapes.clone(); targets = targets.clone();
        }
        @Override public short[] shapes() { return shapes.clone(); }
        @Override public short[] targets() { return targets.clone(); }
    }

    public static Optional<Target> acquire(Scan scan) {
        // The record owns its arrays; they stay alive until this synchronous JNI call returns.
        double[] data = scan0(new double[]{scan.eye.x, scan.eye.y, scan.eye.z,
            scan.orientation.yaw, scan.orientation.pitch}, GeneratedCatalog.SHAPE_CATALOG_VERSION,
            scan.side, scan.reach, scan.shapes, scan.targets);
        if (data.length == 0) return Optional.empty();
        String[] faces = {"down", "up", "north", "south", "west", "east"};
        var components = new ArrayList<List<Vector>>();
        int cursor = 14;
        for (int i = 0; i < (int) data[13]; i++) {
            int vertices = (int) data[cursor++];
            var directions = new ArrayList<Vector>();
            for (int j = 0; j < vertices; j++) {
                directions.add(new Vector(data[cursor], data[cursor + 1], data[cursor + 2]));
                cursor += 3;
            }
            components.add(directions);
        }
        if (cursor != data.length) throw new IllegalStateException("Invalid native target payload");
        return Optional.of(new Target(data[0], data[1], data[2], data[3], data[4], data[5],
            new Block((int) data[6], (int) data[7], (int) data[8]), faces[(int) data[9]],
            new Vector(data[10], data[11], data[12]), components));
    }

    public static AimResult generate(Orientation start, Target target, AimConfig config, double step, long seed) {
        double[] data = aim0(config.modelCode(), new double[]{start.yaw, start.pitch}, target.metrics(),
            encodeComponents(target.components), step, config.minimumPayload(), config.sigmaPayload(),
            config.feedbackPayload(), seed);
        int points = (int) data[0];
        if (points < 0 || 1L + points * 3L > data.length) throw new IllegalStateException("Invalid native aim payload");
        var path = new ArrayList<AimPoint>(points);
        for (int i = 0; i < points; i++) path.add(new AimPoint(data[1 + i * 3], data[2 + i * 3], data[3 + i * 3]));
        var diagnostics = new ArrayList<Double>();
        for (int i = 1 + points * 3; i < data.length; i++) diagnostics.add(data[i]);
        return new AimResult(path, diagnostics);
    }

    static double[] encodeComponents(List<List<Vector>> components) {
        int size = 1;
        for (var component : components) size += 1 + 3 * component.size();
        double[] data = new double[size];
        data[0] = components.size();
        int cursor = 1;
        for (var component : components) {
            data[cursor++] = component.size();
            for (var v : component) { data[cursor++] = v.x; data[cursor++] = v.y; data[cursor++] = v.z; }
        }
        return data;
    }

    private static native int abiVersion0();
    private static native String catalogFingerprint0();
    static native double[] scan0(double[] pose, int version, int side, double reach, short[] shapes, short[] targets);
    private static native double[] aim0(int model, double[] start, double[] target, double[] components,
        double step, double[] minimum, double[] sigma, double[] feedback, long seed);
}
