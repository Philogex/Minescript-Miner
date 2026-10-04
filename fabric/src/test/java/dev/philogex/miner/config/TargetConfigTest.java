package dev.philogex.miner.config;

import static org.junit.jupiter.api.Assertions.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TargetConfigTest {
    @Test void readsLiteralIdsWithBlankLinesAndComments(@TempDir Path directory) throws Exception {
        Path file = directory.resolve("targets.txt");
        Files.writeString(file, "\n# comment\n minecraft:stone \nminecraft:cobblestone # inline comment\nminecraft:stone\n");
        assertEquals(Set.of("minecraft:stone", "minecraft:cobblestone"), TargetConfig.load(file));
    }
    @Test void emptyAndCommentOnlyConfigurationIsEmpty() {
        assertTrue(TargetConfig.parse(List.of("", " ", "# comment", " # another comment")).isEmpty());
    }
}
