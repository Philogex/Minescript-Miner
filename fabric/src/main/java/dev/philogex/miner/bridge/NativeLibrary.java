package dev.philogex.miner.bridge;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

final class NativeLibrary {
    private NativeLibrary() {}

    static void load() {
        String override = System.getProperty("miner.native.path");
        if (override != null) {
            System.load(Path.of(override).toAbsolutePath().toString());
            return;
        }
        String system = System.getProperty("os.name").toLowerCase(Locale.ROOT);
        String os = system.contains("linux") ? "linux" : system.contains("windows") ? "windows" : "macos";
        String arch = System.getProperty("os.arch");
        if (arch.equals("amd64")) arch = "x86_64";
        String name = System.mapLibraryName("miner_jni");
        String resource = "/native/" + os + "-" + arch + "/" + name;
        try (var stream = NativeLibrary.class.getResourceAsStream(resource)) {
            if (stream == null) throw new IOException("No native library for " + os + "-" + arch);
            Path directory = Files.createTempDirectory("minecraft-miner-native-");
            directory.toFile().deleteOnExit();
            Path library = directory.resolve(name);
            Files.copy(stream, library);
            library.toFile().deleteOnExit();
            System.load(library.toAbsolutePath().toString());
        } catch (IOException error) {
            throw new IllegalStateException("Cannot load miner native library", error);
        }
    }
}
