package dev.philogex.miner.config;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class AimConfig {
    private static final String[] MINIMUM = {"fitts_a_ms", "fitts_b_ms", "min_duration_ms", "max_duration_ms", "sample_hz", "correction_probability", "max_corrections"};
    private static final double[] MINIMUM_DEFAULTS = {80, 110, 60, 450, 120, .6, 1};
    private static final String[] SIGMA = {"fitts_a", "fitts_b", "target_width", "undershoot_min", "undershoot_max", "peak_time_ratio", "primary_sigma_min", "primary_sigma_max", "overshoot_prob", "overshoot_min", "overshoot_max", "correction_sigma_min", "correction_sigma_max", "second_correction_prob", "curvature_scale", "ou_theta", "ou_sigma", "tremor_freq_min", "tremor_freq_max", "tremor_amp_min", "tremor_amp_max", "sdn_k", "sample_dt_mean", "gamma_shape"};
    private static final double[] SIGMA_DEFAULTS = {50, 150, 20, .92, .97, .35, .18, .28, .15, 1.02, 1.08, .12, .20, .25, .025, 3.5, 1.2, 8, 12, .15, .55, .04, 7.8, 3.5};
    private static final String[] FEEDBACK = {"feedback_latency_mean_ms", "feedback_latency_stddev_ms", "feedback_latency_min_ms", "feedback_latency_max_ms", "undershoot_width_min", "undershoot_width_max", "overshoot_width_min", "overshoot_width_max", "feedback_position_uncertainty_steps", "safe_margin_steps", "max_corrections"};
    private static final double[] FEEDBACK_DEFAULTS = {100, 15, 60, 160, .05, .25, .05, .25, .25, 1, 3};
    private final String model;
    private final double fallbackStep;
    private final Map<String, Map<String, Double>> sections;

    private AimConfig(String model, double fallbackStep, Map<String, Map<String, Double>> sections) {
        this.model = model; this.fallbackStep = fallbackStep;
        var copy = new HashMap<String, Map<String, Double>>();
        sections.forEach((key, values) -> copy.put(key, Map.copyOf(values)));
        this.sections = Map.copyOf(copy);
        validate();
    }
    public static AimConfig load(Path path) throws IOException { return parse(Files.readAllLines(path)); }
    public static AimConfig defaults() { return parse(List.of()); }
    public static AimConfig fromPayloads(String model, double[] minimum, double[] sigma, double[] feedback) {
        require(minimum.length == 5 && sigma.length == SIGMA.length && feedback.length == FEEDBACK.length, "Invalid config payload lengths");
        var sections = new HashMap<String, Map<String, Double>>();
        var m = defaults(MINIMUM, MINIMUM_DEFAULTS);
        for (int i = 0; i < minimum.length; i++) m.put(MINIMUM[i], minimum[i]);
        sections.put("minimum_jerk", m);
        sections.put("sigmadrift", defaults(SIGMA, sigma));
        sections.put("geometry_feedback_sigmadrift", defaults(FEEDBACK, feedback));
        return new AimConfig(model, .15, sections);
    }
    public static AimConfig parse(List<String> lines) {
        var values = new HashMap<String, Map<String, Double>>();
        values.put("minimum_jerk", defaults(MINIMUM, MINIMUM_DEFAULTS));
        values.put("sigmadrift", defaults(SIGMA, SIGMA_DEFAULTS));
        values.put("geometry_feedback_sigmadrift", defaults(FEEDBACK, FEEDBACK_DEFAULTS));
        String section = "global", model = "minimum_jerk";
        double fallback = .15;
        int number = 0;
        for (String raw : lines) {
            number++;
            String line = raw.split("#", 2)[0].strip();
            if (line.isEmpty()) continue;
            if (line.endsWith("[")) {
                section = line.substring(0, line.length() - 1).strip();
                require(values.containsKey(section), "Unknown aim section at line " + number);
                continue;
            }
            if (line.equals("]")) { section = "global"; continue; }
            var pair = line.split(":", 2);
            require(pair.length == 2, "Expected name: value at line " + number);
            String name = pair[0].strip(), text = pair[1].strip();
            if (section.equals("global") && name.equals("aim_model")) { model = text; continue; }
            if (section.equals("global") && name.equals("fallback_angular_step_deg")) {
                fallback = Double.parseDouble(text); continue;
            }
            String destination = section.equals("global") ? "minimum_jerk" : section;
            require(values.get(destination).containsKey(name), "Unknown aim key at line " + number + ": " + name);
            double value = Double.parseDouble(text);
            if (name.equals("sample_hz") || name.equals("max_corrections"))
                require(value == Math.rint(value) && value >= 0 && value <= Integer.MAX_VALUE, "Invalid integer: " + name);
            values.get(destination).put(name, value);
        }
        return new AimConfig(model, fallback, values);
    }
    private static Map<String, Double> defaults(String[] names, double[] values) {
        var map = new LinkedHashMap<String, Double>();
        for (int i = 0; i < names.length; i++) map.put(names[i], values[i]);
        return map;
    }
    public AimConfig withModel(String model) { return new AimConfig(model, fallbackStep, sections); }
    public int modelCode() {
        return switch (model) {
            case "minimum_jerk" -> 0;
            case "sigmadrift" -> 1;
            case "geometry_feedback_sigmadrift" -> 2;
            default -> throw new IllegalArgumentException("Unsupported aim model: " + model);
        };
    }
    public String model() { return model; }
    public double fallbackStep() { return fallbackStep; }
    /** Named configuration for analysis metadata; the immutable sections remain owned here. */
    public Map<String, Object> metadata() {
        var result = new LinkedHashMap<String, Object>();
        result.put("aim_model", model);
        result.put("fallback_angular_step_deg", fallbackStep);
        sections.forEach((section, values) -> {
            var named = new LinkedHashMap<String, Object>();
            values.forEach((key, value) -> {
                if (key.equals("sample_hz") || key.equals("max_corrections")) named.put(key, value.intValue());
                else named.put(key, value);
            });
            result.put(section, Map.copyOf(named));
        });
        return Map.copyOf(result);
    }
    public double[] minimumPayload() { return payload("minimum_jerk", java.util.Arrays.copyOf(MINIMUM, 5)); }
    public double[] sigmaPayload() { return payload("sigmadrift", SIGMA); }
    public double[] feedbackPayload() { return payload("geometry_feedback_sigmadrift", FEEDBACK); }
    private double[] payload(String section, String[] names) {
        double[] values = new double[names.length];
        for (int i = 0; i < names.length; i++) values[i] = sections.get(section).get(names[i]);
        return values;
    }
    private double value(String section, String name) { return sections.get(section).get(name); }
    private void validate() {
        modelCode();
        require(Double.isFinite(fallbackStep) && fallbackStep > 0, "Invalid fallback step");
        sections.values().forEach(map -> map.values().forEach(v -> require(Double.isFinite(v), "Aim values must be finite")));
        positive("minimum_jerk", "sample_hz");
        require(value("minimum_jerk", "sample_hz") == Math.rint(value("minimum_jerk", "sample_hz")), "Invalid sample_hz");
        require(value("geometry_feedback_sigmadrift", "max_corrections") == Math.rint(value("geometry_feedback_sigmadrift", "max_corrections")), "Invalid max_corrections");
        range("minimum_jerk", "min_duration_ms", "max_duration_ms");
        probability("minimum_jerk", "correction_probability");
        for (String name : List.of("target_width", "sample_dt_mean", "gamma_shape")) positive("sigmadrift", name);
        for (String name : List.of("overshoot_prob", "second_correction_prob")) probability("sigmadrift", name);
        for (String stem : List.of("undershoot", "primary_sigma", "overshoot", "correction_sigma", "tremor_freq", "tremor_amp"))
            range("sigmadrift", stem + "_min", stem + "_max");
        for (String name : FEEDBACK) require(value("geometry_feedback_sigmadrift", name) >= 0, "Negative feedback value: " + name);
        range("geometry_feedback_sigmadrift", "feedback_latency_min_ms", "feedback_latency_max_ms");
        range("geometry_feedback_sigmadrift", "undershoot_width_min", "undershoot_width_max");
        range("geometry_feedback_sigmadrift", "overshoot_width_min", "overshoot_width_max");
        require(value("geometry_feedback_sigmadrift", "max_corrections") <= 64, "Too many corrections");
    }
    private void positive(String section, String name) { require(value(section, name) > 0, "Expected positive " + name); }
    private void probability(String section, String name) { require(value(section, name) >= 0 && value(section, name) <= 1, "Invalid probability " + name); }
    private void range(String section, String lower, String upper) { require(value(section, upper) >= value(section, lower), "Invalid range " + lower); }
    private static void require(boolean valid, String message) { if (!valid) throw new IllegalArgumentException(message); }
    public static double angularStep(double sensitivity) { double f = sensitivity * .6 + .2; return f * f * f * 8 * .15; }
}
