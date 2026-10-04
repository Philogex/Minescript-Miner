package dev.philogex.miner.runtime;

import dev.philogex.miner.bridge.NativeBridge;

import static dev.philogex.miner.bridge.NativeBridge.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

class ScanCaptureTest {
    private static final ScanCapture.Sample AIR = new ScanCapture.Sample("minecraft:air", (short) 0, 0);
    private static final ScanCapture.Sample STONE = new ScanCapture.Sample("minecraft:stone", (short) 1, 11);
    private static class Source implements ScanCapture.BlockSource {
        final Map<Block, ScanCapture.Sample> samples;
        final List<Block> reads = new ArrayList<>();
        Block unloaded;
        Source(Map<Block, ScanCapture.Sample> samples) { this.samples = samples; }
        public boolean loaded(Block block) { return !block.equals(unloaded); }
        public ScanCapture.Sample read(Block block) { reads.add(block); return samples.getOrDefault(block, AIR); }
    }
    @Test void preservesCubeLayoutAndFillsUnreachableBlocksWithAir() {
        var source = new Source(Map.of(
            new Block(9, 63, -5), STONE,
            new Block(10, 63, -5), new ScanCapture.Sample("minecraft:oak_slab", (short) 3, 12),
            new Block(9, 63, -4), new ScanCapture.Sample("minecraft:unknown_block", (short) 1, 13),
            new Block(9, 64, -5), new ScanCapture.Sample("minecraft:dirt", (short) 1, 14),
            new Block(11, 65, -3), STONE));
        var captured = ScanCapture.capture(new Vector(10.1, 64.2, -3.7), new Orientation(90, 10),
            1, Set.of("minecraft:stone"), source).orElseThrow();
        var scan = captured.scan();
        assertEquals(3, scan.side());
        assertEquals(new Vector(10.1, 64.2, -3.7), scan.eye());
        assertEquals(new Orientation(90, 10), scan.orientation());
        assertEquals(27, scan.shapes().length);
        assertEquals(1, scan.shapes()[0]);
        assertEquals(3, scan.shapes()[1]); // X advances first.
        assertEquals(1, scan.shapes()[3]); // Then Z.
        assertEquals(1, scan.shapes()[9]); // Then Y.
        assertEquals(0, scan.shapes()[26]);
        assertArrayEquals(new short[]{0}, scan.targets());
        assertEquals(12, captured.states().get(new Block(10, 63, -5)));
        assertFalse(source.reads.contains(new Block(11, 65, -3)));
        assertFalse(captured.states().containsKey(new Block(11, 65, -3)));
        assertThrows(UnsupportedOperationException.class, () -> captured.states().clear());
    }
    @Test void usesBlockAabbDistanceAndIncludesReachBoundary() {
        var source = new Source(Map.of(new Block(2, 0, 0), STONE, new Block(-2, 0, 0), STONE));
        var captured = ScanCapture.capture(new Vector(.9, .5, .5), new Orientation(0, 0),
            1.1, Set.of("minecraft:stone"), source).orElseThrow();
        assertEquals(5, captured.scan().side());
        assertTrue(source.reads.contains(new Block(2, 0, 0)));
        assertFalse(source.reads.contains(new Block(-2, 0, 0)));
        assertTrue(captured.states().containsKey(new Block(2, 0, 0)));
    }
    @Test void targetsMatchLiteralRegistryIdsAcrossBlockStates() {
        var source = new Source(Map.of(
            new Block(0, 0, 1), STONE,
            new Block(1, 0, 0), new ScanCapture.Sample("minecraft:oak_slab", (short) 3, 12),
            new Block(0, 1, 0), new ScanCapture.Sample("minecraft:stone_bricks", (short) 1, 13)));
        var captured = ScanCapture.capture(new Vector(.5, .5, .5), new Orientation(0, 0),
            .5, Set.of("minecraft:stone", "minecraft:oak_slab"), source).orElseThrow();
        assertArrayEquals(new short[]{14, 16}, captured.scan().targets());
        assertEquals(3, captured.scan().shapes()[14]);
        assertEquals(1, captured.scan().shapes()[22]);
        assertEquals(13, captured.states().get(new Block(0, 1, 0)));
    }
    @Test void unloadedBlockRejectsTheWholeSnapshot() {
        var source = new Source(Map.of(new Block(0, 0, 1), STONE));
        source.unloaded = new Block(0, 0, 1);
        assertTrue(ScanCapture.capture(new Vector(.5, .5, .5), new Orientation(0, 0),
            .5, Set.of("minecraft:stone"), source).isEmpty());
        assertFalse(source.reads.contains(source.unloaded));
    }
    @Test void emptyTargetConfigurationSkipsWorldAccess() {
        var source = new ScanCapture.BlockSource() {
            public boolean loaded(Block block) { fail("World queried without configured targets"); return false; }
            public ScanCapture.Sample read(Block block) { fail("World read without configured targets"); return AIR; }
        };
        assertTrue(ScanCapture.capture(new Vector(.5, .5, .5), new Orientation(0, 0),
            .5, Set.of(), source).isEmpty());
    }
    @Test void noMatchingBlocksProducesNoPlanningSnapshot() {
        var source = new Source(Map.of());
        assertTrue(ScanCapture.capture(new Vector(.5, .5, .5), new Orientation(0, 0),
            .5, Set.of("minecraft:stone"), source).isEmpty());
        assertFalse(source.reads.isEmpty());
    }
    @Test void oversizedCubeIsRejectedBeforeReadingOrPackingUint16Indices() {
        var source = new Source(Map.of());
        assertThrows(IllegalArgumentException.class, () -> ScanCapture.capture(
            new Vector(.5, .5, .5), new Orientation(0, 0), 20, Set.of("minecraft:stone"), source));
        assertTrue(source.reads.isEmpty());
    }
}
