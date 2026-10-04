package dev.philogex.miner.fabric.mixin;

import dev.philogex.miner.fabric.MinerClient;
import net.minecraft.client.MouseHandler;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(MouseHandler.class)
public abstract class MouseHandlerMixin {
    @Inject(method = "handleAccumulatedMovement", at = @At("TAIL"))
    private void miner$applyAim(CallbackInfo info) { MinerClient.frame(); }
}
