package dev.philogex.miner.catalog;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

public final class ShapeCatalog {
    private static final Set<String> EMPTY = new HashSet<>();
    private static final Map<String, List<String>> RELEVANT = new HashMap<>();
    private static final Map<String, Integer> SHAPES = new HashMap<>();
    static {
        try (var stream = ShapeCatalog.class.getResourceAsStream("/catalog/block_shapes.tsv")) {
            if (stream == null) throw new IllegalStateException("Missing shape catalog");
            try (var reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
                for (String line; (line = reader.readLine()) != null;) {
                    if (line.startsWith("#") || line.isBlank()) continue;
                    String[] parts = line.split("\t", -1);
                    switch (parts[0]) {
                        case "empty" -> EMPTY.add(parts[1]);
                        case "properties" -> RELEVANT.put(parts[1], parts[2].isEmpty() ? List.of() : List.of(parts[2].split(",")));
                        case "shape" -> SHAPES.put(parts[1], Integer.parseInt(parts[2]));
                        default -> throw new IllegalStateException("Invalid catalog row");
                    }
                }
            }
        } catch (Exception error) { throw new ExceptionInInitializerError(error); }
    }
    private ShapeCatalog() {}

    public static int shapeId(String block, Map<String, String> state) {
        if (EMPTY.contains(block)) return GeneratedCatalog.SHAPE_EMPTY;
        var relevant = RELEVANT.get(block);
        if (relevant == null) return GeneratedCatalog.SHAPE_FULL_CUBE;
        var selected = new TreeMap<String, String>();
        for (var name : relevant) {
            if (!state.containsKey(name)) return GeneratedCatalog.SHAPE_FULL_CUBE;
            selected.put(name, state.get(name));
        }
        String properties = String.join(",", selected.entrySet().stream().map(e -> e.getKey() + "=" + e.getValue()).toList());
        return SHAPES.getOrDefault(block + "[" + properties + "]", GeneratedCatalog.SHAPE_FULL_CUBE);
    }

    public static int shapeId(String blockState) {
        if (blockState == null || blockState.isBlank()) return GeneratedCatalog.SHAPE_EMPTY;
        String text = blockState.strip();
        int bracket = text.indexOf('[');
        if (bracket < 0) return shapeId(text, Map.of());
        var properties = new HashMap<String, String>();
        if (text.endsWith("]")) {
            for (var property : text.substring(bracket + 1, text.length() - 1).split(",")) {
                var pair = property.split("=", 2);
                if (pair.length == 2) properties.put(pair[0].strip(), pair[1].strip());
            }
        }
        return shapeId(text.substring(0, bracket).strip(), properties);
    }
}
