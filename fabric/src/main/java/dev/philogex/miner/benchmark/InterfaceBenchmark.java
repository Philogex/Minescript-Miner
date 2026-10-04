package dev.philogex.miner.benchmark;

import dev.philogex.miner.bridge.NativeBridge;
import dev.philogex.miner.catalog.GeneratedCatalog;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.function.IntSupplier;

/** Offline API benchmark; deliberately does not initialize Minecraft. */
public final class InterfaceBenchmark {
    private static volatile int consumed;
    private InterfaceBenchmark() {}
    public static void main(String[] args) throws Exception {
        if (args.length < 2 || args.length > 4) throw new IllegalArgumentException("reference-directory output.csv [samples=1000] [warmup=200]");
        int samples = args.length > 2 ? Integer.parseInt(args[2]) : 1000;
        int warmup = args.length > 3 ? Integer.parseInt(args[3]) : 200;
        if (samples < 1 || warmup < 0) throw new IllegalArgumentException("Invalid benchmark counts");
        var output = new StringBuilder("runtime,fixture,operation,iteration,elapsed_ns\n");
        try (var files = Files.list(Path.of(args[0]))) {
            for (var path : files.filter(p -> p.toString().endsWith(".properties")).sorted().toList()) {
                var fixture = ReferenceFixture.read(path);
                measure(fixture.name(), "scan", () -> NativeBridge.acquire(fixture.scan()).isPresent() ? 1 : 0, warmup, samples, output);
                var target = NativeBridge.acquire(fixture.scan());
                if (target.isEmpty()) continue;
                for (String model : List.of("minimum_jerk", "sigmadrift", "geometry_feedback_sigmadrift")) {
                    var config = fixture.config(model);
                    measure(fixture.name(), model,
                        () -> NativeBridge.generate(fixture.scan().orientation(), target.get(), config, fixture.step(), fixture.seed()).points().size(),
                        warmup, samples, output);
                }
            }
        }
        Path file = Path.of(args[1]); Files.createDirectories(file.toAbsolutePath().getParent());
        Files.writeString(file, output);
        Path artifact = Path.of(InterfaceBenchmark.class.getProtectionDomain().getCodeSource().getLocation().toURI());
        String artifactHash = Files.isRegularFile(artifact)
            ? java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(artifact))) : "classes-directory";
        Files.writeString(Path.of(args[1] + ".metadata.txt"),
            "runtime=" + System.getProperty("java.runtime.version") + "\nos=" + System.getProperty("os.name")
            + "\narch=" + System.getProperty("os.arch") + "\nsamples=" + samples + "\nwarmup=" + warmup
            + "\nartifact_sha256=" + artifactHash + "\ncatalog_sha256=" + GeneratedCatalog.SHAPE_CATALOG_SHA256 + "\n");
        System.out.println("Offline Java benchmark: " + file);
    }
    private static void measure(String fixture, String operation, IntSupplier call, int warmup, int samples, StringBuilder output) {
        for (int i = 0; i < warmup; i++) consumed = call.getAsInt();
        long[] times = new long[samples];
        for (int i = 0; i < samples; i++) {
            long start = System.nanoTime(); int value = call.getAsInt();
            times[i] = System.nanoTime() - start; consumed = value;
        }
        for (int i = 0; i < samples; i++) output.append("java,").append(fixture).append(',').append(operation)
            .append(',').append(i).append(',').append(times[i]).append('\n');
    }
}
