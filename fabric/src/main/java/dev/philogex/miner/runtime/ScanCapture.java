package dev.philogex.miner.runtime;

import dev.philogex.miner.bridge.NativeBridge;
import dev.philogex.miner.catalog.GeneratedCatalog;

import static dev.philogex.miner.bridge.NativeBridge.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/** Packs world samples into the scanner's x/z/y layout without game dependencies. */
public final class ScanCapture {
    private ScanCapture() {}
    public record Sample(String blockId, short shapeId, int stateId) {}
    public interface BlockSource {
        boolean loaded(Block block);
        Sample read(Block block);
    }
    public record Capture(Scan scan, Map<Block, Integer> states) {
        public Capture { states = Map.copyOf(states); }
    }

    public static Optional<Capture> capture(Vector eye, Orientation orientation, double reach,
                                            Set<String> targets, BlockSource source) {
        if (!Double.isFinite(reach) || reach <= 0 || Math.ceil(reach) > (GeneratedCatalog.MAX_CUBE_SIDE - 1) / 2)
            throw new IllegalArgumentException("Reach exceeds the supported scan cube");
        if (targets.isEmpty()) return Optional.empty();
        int half = (int) Math.ceil(reach), side = 2 * half + 1;
        int minX = (int) Math.floor(eye.x()) - half;
        int minY = (int) Math.floor(eye.y()) - half;
        int minZ = (int) Math.floor(eye.z()) - half;
        short[] shapes = new short[side * side * side];
        var targetIndices = new ArrayList<Short>();
        var states = new HashMap<Block, Integer>();
        for (int y = 0; y < side; y++) for (int z = 0; z < side; z++) for (int x = 0; x < side; x++) {
            var block = new Block(minX + x, minY + y, minZ + z);
            if (!source.loaded(block)) return Optional.empty();
            double dx = Math.max(Math.max(block.x() - eye.x(), 0), eye.x() - block.x() - 1);
            double dy = Math.max(Math.max(block.y() - eye.y(), 0), eye.y() - block.y() - 1);
            double dz = Math.max(Math.max(block.z() - eye.z(), 0), eye.z() - block.z() - 1);
            if (dx * dx + dy * dy + dz * dz > reach * reach) continue;
            var sample = source.read(block);
            int index = x + z * side + y * side * side;
            shapes[index] = sample.shapeId();
            states.put(block, sample.stateId());
            if (targets.contains(sample.blockId())) targetIndices.add((short) index);
        }
        if (targetIndices.isEmpty()) return Optional.empty();
        short[] indices = new short[targetIndices.size()];
        for (int i = 0; i < indices.length; i++) indices[i] = targetIndices.get(i);
        return Optional.of(new Capture(new Scan(eye, orientation, side, reach, shapes, indices), states));
    }
}
