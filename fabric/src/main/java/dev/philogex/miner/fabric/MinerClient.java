package dev.philogex.miner.fabric;

import com.mojang.blaze3d.platform.InputConstants;
import dev.philogex.miner.config.AimConfig;
import dev.philogex.miner.runtime.MinerController;
import dev.philogex.miner.bridge.NativeBridge;
import dev.philogex.miner.catalog.ShapeCatalog;
import dev.philogex.miner.runtime.ScanCapture;
import dev.philogex.miner.config.TargetConfig;
import static dev.philogex.miner.bridge.NativeBridge.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.keymapping.v1.KeyMappingHelper;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.Identifier;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.HitResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class MinerClient implements ClientModInitializer {
    private static final Logger LOG = LoggerFactory.getLogger("minecraft_miner");
    private static MinerController controller;
    private static final double REACH = 4.5;

    @Override public void onInitializeClient() {
        try {
            Path directory = FabricLoader.getInstance().getConfigDir().resolve("minecraft-miner");
            Files.createDirectories(directory);
            for (String name : java.util.List.of("targets.txt", "aim_config.txt")) {
                Path file = directory.resolve(name);
                if (!Files.exists(file)) {
                    try (var stream = MinerClient.class.getResourceAsStream("/defaults/" + name)) {
                        if (stream == null) throw new IllegalStateException("Missing default " + name);
                        Files.copy(stream, file);
                    }
                }
            }
            AimConfig config = AimConfig.load(directory.resolve("aim_config.txt"));
            Set<String> targets = TargetConfig.load(directory.resolve("targets.txt"));
            var world = new MinecraftAccess(targets);
            var worker = Executors.newSingleThreadExecutor(task -> {
                var thread = new Thread(task, "minecraft-miner-solver"); thread.setDaemon(true); return thread;
            });
            controller = new MinerController(world, snapshot -> {
                long begin = System.nanoTime();
                var target = NativeBridge.acquire(snapshot.scan());
                long solved = System.nanoTime();
                var path = target.map(value -> NativeBridge.generate(snapshot.scan().orientation(), value,
                    config, snapshot.angularStep(), ThreadLocalRandom.current().nextLong()).points()).orElse(java.util.List.of());
                long end = System.nanoTime();
                LOG.debug("scan_ms={} aim_ms={}", (solved - begin) / 1e6, (end - solved) / 1e6);
                return new MinerController.Plan(snapshot, target, path, solved - begin, end - solved);
            }, worker);
            var category = KeyMapping.Category.register(Identifier.fromNamespaceAndPath("minecraft_miner", "general"));
            var toggle = KeyMappingHelper.registerKeyMapping(new KeyMapping("key.minecraft_miner.toggle",
                InputConstants.Type.KEYBOARD, InputConstants.KEY_O, category));
            ClientTickEvents.END_CLIENT_TICK.register(client -> {
                world.updateWorld();
                if (client.player == null || client.level == null || client.gui.screen() != null || !client.isWindowActive()) {
                    if (controller.active()) controller.setActive(false);
                    return;
                }
                while (toggle.consumeClick()) {
                    controller.setActive(!controller.active());
                    client.gui.hud.getChat().addClientSystemMessage(Component.literal("Miner active: " + controller.active()));
                }
                world.angularStep = AimConfig.angularStep(client.options.sensitivity().get());
                controller.tick(System.nanoTime());
            });
            ClientLifecycleEvents.CLIENT_STOPPING.register(client -> {
                controller.setActive(false); worker.shutdownNow();
            });
            LOG.info("Miner ready: O toggles mining; configuration in {}", directory);
        } catch (Exception error) { LOG.error("Cannot initialize Minecraft Miner", error); }
    }

    // Called after vanilla mouse processing, on the client thread. The method
    // runs even when no mouse delta was accumulated; no timer thread writes game state.
    public static void frame() {
        if (controller == null) return;
        var client = Minecraft.getInstance();
        if (client.player == null || client.level == null || client.gui.screen() != null || !client.isWindowActive()) {
            if (controller.active()) controller.setActive(false);
            return;
        }
        controller.frame(System.nanoTime());
    }

    private static final class MinecraftAccess implements MinerController.WorldAccess {
        private final Set<String> targets;
        private final Map<BlockState, Short> shapeCache = new IdentityHashMap<>();
        private Object level;
        private long worldId;
        private double angularStep = .15;
        MinecraftAccess(Set<String> targets) { this.targets = targets; }
        void updateWorld() {
            Object current = Minecraft.getInstance().level;
            if (current != level) {
                level = current; worldId++; shapeCache.clear();
                if (controller.active()) controller.setActive(false);
            }
        }
        @Override public Optional<MinerController.Snapshot> capture(long now) {
            var client = Minecraft.getInstance();
            if (client.level == null || client.player == null || targets.isEmpty()) return Optional.empty();
            var eye = client.player.getEyePosition();
            return ScanCapture.capture(new Vector(eye.x, eye.y, eye.z),
                new Orientation(client.player.getYRot(), client.player.getXRot()), REACH, targets,
                new ScanCapture.BlockSource() {
                    public boolean loaded(Block block) {
                        return client.level.hasChunkAt(new BlockPos(block.x(), block.y(), block.z()));
                    }
                    public ScanCapture.Sample read(Block block) {
                        var state = client.level.getBlockState(new BlockPos(block.x(), block.y(), block.z()));
                        return new ScanCapture.Sample(BuiltInRegistries.BLOCK.getKey(state.getBlock()).toString(),
                            shapeCache.computeIfAbsent(state, MinecraftAccess::encode), net.minecraft.world.level.block.Block.getId(state));
                    }
                }).map(capture -> new MinerController.Snapshot(capture.scan(), capture.states(), worldId, now, angularStep));
        }
        private static short encode(BlockState state) {
            var values = new HashMap<String, String>();
            state.getValues().forEach(value -> values.put(value.property().getName(), value.valueName()));
            return (short) ShapeCatalog.shapeId(BuiltInRegistries.BLOCK.getKey(state.getBlock()).toString(), values);
        }
        @Override public boolean valid(MinerController.Snapshot snapshot, Target target) {
            var client = Minecraft.getInstance();
            if (snapshot.worldId() != worldId || client.level != level || client.player == null || client.level == null) return false;
            var eye = client.player.getEyePosition(); var expected = snapshot.scan().eye();
            double dx = eye.x - expected.x(), dy = eye.y - expected.y(), dz = eye.z - expected.z();
            if (dx * dx + dy * dy + dz * dz > 1e-6) return false;
            var block = target.block(); var position = new BlockPos(block.x(), block.y(), block.z());
            return client.level.hasChunkAt(position) && java.util.Objects.equals(snapshot.states().get(block),
                net.minecraft.world.level.block.Block.getId(client.level.getBlockState(position)));
        }
        @Override public boolean hitting(Target target) {
            var client = Minecraft.getInstance();
            if (client.player == null || client.level == null) return false;
            var eye = client.player.getEyePosition();
            var end = eye.add(client.player.getViewVector(1.0F).scale(REACH));
            var hit = client.level.clip(new ClipContext(eye, end, ClipContext.Block.OUTLINE, ClipContext.Fluid.NONE, client.player));
            var block = target.block();
            var position = new BlockPos(block.x(), block.y(), block.z());
            // Normal attack consumes Minecraft's picked hit, which may be an
            // entity even when the independent block-only ray reaches this block.
            return hit.getType() == HitResult.Type.BLOCK && hit.getBlockPos().equals(position)
                && client.hitResult instanceof BlockHitResult picked && picked.getType() == HitResult.Type.BLOCK
                && picked.getBlockPos().equals(position);
        }
        @Override public void orient(double yaw, double pitch) {
            var player = Minecraft.getInstance().player;
            if (player != null) { player.setYRot((float) yaw); player.setXRot((float) pitch); }
        }
        @Override public void attack(boolean pressed) { Minecraft.getInstance().options.keyAttack.setDown(pressed); }
        @Override public void failed(Throwable error) {
            LOG.error("Miner calculation failed", error);
            var player = Minecraft.getInstance().player;
            if (player != null) Minecraft.getInstance().gui.hud.getChat().addClientSystemMessage(Component.literal("Miner stopped: " + error.getMessage()));
        }
    }
}
