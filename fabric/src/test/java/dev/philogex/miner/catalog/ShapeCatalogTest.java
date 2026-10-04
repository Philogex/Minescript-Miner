package dev.philogex.miner.catalog;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

class ShapeCatalogTest {
    @Test void ignoresIrrelevantPropertiesAndTheirOrder() {
        assertEquals(ShapeCatalog.shapeId("minecraft:oak_stairs[facing=east,half=top,shape=inner_left]"),
            ShapeCatalog.shapeId("minecraft:oak_stairs[waterlogged=true,shape=inner_left,half=top,facing=east]"));
        assertEquals(1, ShapeCatalog.shapeId("minecraft:oak_stairs"));
        assertEquals(0, ShapeCatalog.shapeId((String)null));
    }
}
