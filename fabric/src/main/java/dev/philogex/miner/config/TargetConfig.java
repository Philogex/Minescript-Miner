package dev.philogex.miner.config;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public final class TargetConfig {
    private TargetConfig() {}
    public static Set<String> load(Path path) throws IOException { return parse(Files.readAllLines(path)); }
    public static Set<String> parse(List<String> lines) {
        var targets = new HashSet<String>();
        for (var raw : lines) {
            String line = raw.split("#", 2)[0].strip();
            if (!line.isEmpty()) targets.add(line);
        }
        return Set.copyOf(targets);
    }
}
