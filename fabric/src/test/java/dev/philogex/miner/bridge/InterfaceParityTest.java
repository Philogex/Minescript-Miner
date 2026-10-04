package dev.philogex.miner.bridge;

import dev.philogex.miner.benchmark.ReferenceFixture;
import dev.philogex.miner.catalog.GeneratedCatalog;
import dev.philogex.miner.catalog.ShapeCatalog;

import static dev.philogex.miner.bridge.NativeBridge.*;
import static org.junit.jupiter.api.Assertions.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class InterfaceParityTest {
    static Path referenceDirectory() {
        return Path.of(System.getProperty("miner.project.dir"), "fabric/src/test/resources/reference");
    }
    static Stream<Path> fixtures() throws Exception {
        try (var files = Files.list(referenceDirectory())) {
            return files.filter(path -> path.toString().endsWith(".properties")).sorted().toList().stream();
        }
    }
    @ParameterizedTest(name = "{0}") @MethodSource("fixtures")
    void matchesRecordedResults(Path path) throws Exception {
        var fixture = ReferenceFixture.read(path);
        var result = NativeBridge.acquire(fixture.scan());
        assertEquals(Boolean.parseBoolean(fixture.values().getProperty("found")), result.isPresent());
        if (result.isEmpty()) return;
        var target = result.get();
        assertArrayEquals(fixture.numbers("metrics"), target.metrics(), 1e-12);
        assertArrayEquals(fixture.numbers("block"), new double[]{target.block().x(), target.block().y(), target.block().z()}, 0);
        assertEquals(fixture.values().getProperty("face"), target.face());
        assertArrayEquals(fixture.numbers("hit"), new double[]{target.hit().x(), target.hit().y(), target.hit().z()}, 1e-12);
        assertArrayEquals(fixture.numbers("components"), NativeBridge.encodeComponents(target.components()), 1e-12);
        for (String model : List.of("minimum_jerk", "sigmadrift", "geometry_feedback_sigmadrift")) {
            var aim = NativeBridge.generate(fixture.scan().orientation(), target, fixture.config(model), fixture.step(), fixture.seed());
            double[] points = aim.points().stream().flatMapToDouble(p -> java.util.stream.DoubleStream.of(p.yaw(), p.pitch(), p.tMs())).toArray();
            // The references use libstdc++ distributions. MSVC's normal/gamma
            // distributions produce different valid paths for the same seed.
            if (System.getProperty("os.name").equals("Linux") || model.equals("minimum_jerk")) {
                assertArrayEquals(fixture.numbers(model + ".points"), points, 1e-12, model + " path");
                assertArrayEquals(fixture.numbers(model + ".diagnostics"), aim.diagnostics().stream().mapToDouble(Double::doubleValue).toArray(), 1e-12, model + " diagnostics");
            }
            assertEquals(aim, NativeBridge.generate(fixture.scan().orientation(), target,
                fixture.config(model), fixture.step(), fixture.seed()), model + " repeatability");
            assertFalse(aim.points().isEmpty());
            for (var point : aim.points()) {
                assertTrue(Double.isFinite(point.yaw()) && Double.isFinite(point.pitch()) && Double.isFinite(point.tMs()));
                assertTrue(point.pitch() >= -90 && point.pitch() <= 90);
                assertTrue(point.tMs() >= 0);
            }
            assertEquals(model.equals("geometry_feedback_sigmadrift") ? 27 : 0, aim.diagnostics().size());
        }
    }
    @Test void catalogMatchesAllFrozenPythonCases() throws Exception {
        for (var line : Files.readAllLines(referenceDirectory().resolve("catalog.tsv"))) {
            var parts = line.split("\t");
            assertEquals(Integer.parseInt(parts[1]), ShapeCatalog.shapeId(parts[0]), parts[0]);
        }
    }
    @Test void generatedCatalogMatchesSourceFingerprint() throws Exception {
        byte[] bytes = Files.readAllBytes(Path.of(System.getProperty("miner.project.dir"), "catalog/shape_catalog.json"));
        String fingerprint = java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(bytes));
        assertEquals(fingerprint, GeneratedCatalog.SHAPE_CATALOG_SHA256);
    }
    @Test void vendoredBoostIsPinnedAndDoesNotShadowStandardHeader() throws Exception {
        Path root = Path.of(System.getProperty("miner.project.dir"), "third_party/boost");
        assertEquals("1.91.0", Files.readString(root.resolve("BOOST_VERSION")).strip());
        assertFalse(Files.exists(root.resolve("VERSION")), "VERSION shadows <version> on Windows");
        for (String file : List.of("LICENSE_1_0.txt", "boost/multiprecision/cpp_int.hpp",
                "boost/rational.hpp", "boost/integer/common_factor_rt.hpp"))
            assertTrue(Files.isRegularFile(root.resolve(file)), file);
        assertTrue(Files.readString(root.resolve("boost/version.hpp")).contains("#define BOOST_VERSION 109100"));
    }
    @Test void rejectsInvalidNativeInputs() {
        double[] pose = {.5, .5, .5, 0, 0};
        short[] shapes = new short[27];
        assertThrows(IllegalArgumentException.class, () -> NativeBridge.scan0(pose, 2, 3, 4.8, shapes, new short[0]));
        assertThrows(IllegalArgumentException.class, () -> NativeBridge.scan0(pose, 3, 3, 4.8, shapes, new short[]{27}));
        shapes[0] = (short) 65535;
        assertThrows(IllegalArgumentException.class, () -> NativeBridge.scan0(pose, 3, 3, 4.8, shapes, new short[0]));
        assertThrows(IllegalArgumentException.class, () -> NativeBridge.scan0(new double[]{Double.NaN, 0, 0, 0, 0}, 3, 3, 4.8, new short[27], new short[0]));
    }
    @Test void snapshotOwnsItsPayload() {
        short[] shapes = new short[27]; shapes[16] = 1;
        var scan = new Scan(new Vector(.5, .5, .5), new Orientation(0, 0), 3, 4.8, shapes, new short[]{16});
        shapes[16] = 0; scan.shapes()[16] = 0;
        assertTrue(NativeBridge.acquire(scan).isPresent());
    }
    @Test void rejectsCubeOverflowAndMismatchedBuffersAtJniBoundary() {
        double[] pose = {.5, .5, .5, 0, 0};
        assertThrows(IllegalArgumentException.class, () -> NativeBridge.scan0(pose, 3, 41, 20, new short[0], new short[0]));
        assertThrows(IllegalArgumentException.class, () -> NativeBridge.scan0(pose, 3, 3, 4.8, new short[26], new short[0]));
        assertThrows(IllegalArgumentException.class, () -> NativeBridge.scan0(new double[4], 3, 3, 4.8, new short[27], new short[0]));
    }
    @Test void rejectsInvalidReachAtJniBoundary() {
        for (double reach : new double[]{0, -1, Double.NaN, Double.POSITIVE_INFINITY})
            assertThrows(IllegalArgumentException.class, () -> NativeBridge.scan0(
                new double[]{.5, .5, .5, 0, 0}, 3, 3, reach, new short[27], new short[0]));
    }
}
