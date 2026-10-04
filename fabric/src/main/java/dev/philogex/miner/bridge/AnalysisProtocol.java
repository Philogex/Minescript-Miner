package dev.philogex.miner.bridge;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/** Version 1: big-endian length-prefixed frames containing tagged values.
 * Tags: null=0, boolean=1, double=2, int64=3, UTF-8=4, list=5, string-keyed map=6.
 * Collection and string lengths are int32. Each frame contains exactly one value.
 */
final class AnalysisProtocol {
    static final int VERSION = 1;
    private static final int MAX_FRAME = 16 * 1024 * 1024;
    private AnalysisProtocol() {}

    static Object readFrame(DataInputStream input) throws IOException {
        int first = input.read();
        if (first < 0) return null;
        int size = (first << 24) | (input.readUnsignedByte() << 16)
            | (input.readUnsignedByte() << 8) | input.readUnsignedByte();
        if (size < 1 || size > MAX_FRAME) throw new IOException("Invalid frame length: " + size);
        byte[] bytes = input.readNBytes(size);
        if (bytes.length != size) throw new EOFException("Incomplete analysis frame");
        var frame = new DataInputStream(new ByteArrayInputStream(bytes));
        Object value = read(frame, 0);
        if (frame.available() != 0) throw new IOException("Trailing frame data");
        return value;
    }

    static void writeFrame(DataOutputStream output, Object value) throws IOException {
        var bytes = new ByteArrayOutputStream();
        write(new DataOutputStream(bytes), value);
        if (bytes.size() > MAX_FRAME) throw new IOException("Analysis response too large");
        output.writeInt(bytes.size()); bytes.writeTo(output); output.flush();
    }

    private static int length(DataInputStream input) throws IOException {
        int size = input.readInt();
        if (size < 0 || size > input.available()) throw new IOException("Invalid value length");
        return size;
    }

    private static Object read(DataInputStream input, int depth) throws IOException {
        if (depth > 32) throw new IOException("Analysis nesting too deep");
        return switch (input.readUnsignedByte()) {
            case 0 -> null;
            case 1 -> input.readBoolean();
            case 2 -> input.readDouble();
            case 3 -> input.readLong();
            case 4 -> new String(input.readNBytes(length(input)), StandardCharsets.UTF_8);
            case 5 -> {
                int size = length(input);
                var list = new ArrayList<Object>(size);
                for (int i = 0; i < size; i++) list.add(read(input, depth + 1));
                yield list;
            }
            case 6 -> {
                int size = length(input);
                var map = new LinkedHashMap<String, Object>();
                for (int i = 0; i < size; i++) {
                    Object key = read(input, depth + 1);
                    if (!(key instanceof String text)) throw new IOException("Expected string key");
                    map.put(text, read(input, depth + 1));
                }
                yield map;
            }
            default -> throw new IOException("Unknown analysis value tag");
        };
    }

    private static void write(DataOutputStream output, Object value) throws IOException {
        if (value == null) output.writeByte(0);
        else if (value instanceof Boolean b) { output.writeByte(1); output.writeBoolean(b); }
        else if (value instanceof Float || value instanceof Double) { output.writeByte(2); output.writeDouble(((Number)value).doubleValue()); }
        else if (value instanceof Number n) { output.writeByte(3); output.writeLong(n.longValue()); }
        else if (value instanceof String text) {
            byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
            output.writeByte(4); output.writeInt(bytes.length); output.write(bytes);
        } else if (value instanceof List<?> list) {
            output.writeByte(5); output.writeInt(list.size());
            for (var item : list) write(output, item);
        } else if (value instanceof Map<?, ?> map) {
            output.writeByte(6); output.writeInt(map.size());
            for (var entry : map.entrySet()) { write(output, entry.getKey()); write(output, entry.getValue()); }
        } else throw new IOException("Unsupported analysis value: " + value.getClass());
    }
}
