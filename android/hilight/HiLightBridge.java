import android.content.Context;
import android.hardware.lights.Light;
import android.hardware.lights.LightState;
import android.hardware.lights.LightsManager;
import android.hardware.lights.LightsRequest;
import android.os.Looper;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Minimal stdin-driven bridge to Android 17 LightsManager for Pixel 11 Pro HiLight (8 RGB LEDs).
 * Runs under adb shell (com.android.shell, which holds CONTROL_DEVICE_LIGHTS).
 *
 * Stdin commands:
 *   LIST
 *   SOLID <RRGGBB>
 *   SEG <RRGGBB> ... (up to 8 segments)
 *   OFF
 *   QUIT
 */
public final class HiLightBridge {
    private static Context getSystemContext() throws Exception {
        Class<?> atClass = Class.forName("android.app.ActivityThread");
        Method systemMain = atClass.getMethod("systemMain");
        Object at = systemMain.invoke(null);
        Method getCtx = atClass.getMethod("getSystemContext");
        return (Context) getCtx.invoke(at);
    }

    private static int parseColor(String hex) {
        String clean = hex.startsWith("#") ? hex.substring(1) : hex;
        long val = Long.parseLong(clean, 16);
        if (clean.length() <= 6) {
            return (int) (0xFF000000L | (val & 0xFFFFFFL));
        }
        return (int) val;
    }

    public static void main(String[] args) throws Exception {
        Looper.prepareMainLooper();
        Context ctx = getSystemContext();
        LightsManager lm = ctx.getSystemService(LightsManager.class);
        if (lm == null) {
            System.err.println("ERR: LightsManager unavailable");
            System.exit(1);
        }

        List<Light> allLights = lm.getLights();
        List<Light> ringLights = new ArrayList<>();
        for (Light l : allLights) {
            if (l.hasRgbControl() && l.getId() > 0) {
                ringLights.add(l);
            }
        }
        if (ringLights.isEmpty()) {
            for (Light l : allLights) {
                if (l.hasRgbControl()) {
                    ringLights.add(l);
                }
            }
        }
        ringLights.sort(Comparator.comparingInt(Light::getOrdinal));

        System.out.println("READY lights=" + ringLights.size());
        for (Light l : ringLights) {
            System.out.println("LIGHT id=" + l.getId() + " ordinal=" + l.getOrdinal()
                    + " type=" + l.getType() + " rgb=" + l.hasRgbControl()
                    + " minPeriod=" + l.getMinUpdatePeriodMillis());
        }
        System.out.flush();

        try (LightsManager.LightsSession session = lm.openSession();
             BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] parts = line.split("\\s+");
                String cmd = parts[0].toUpperCase();

                if ("QUIT".equals(cmd)) {
                    break;
                } else if ("OFF".equals(cmd)) {
                    LightsRequest.Builder rb = new LightsRequest.Builder();
                    LightState off = new LightState.Builder().setColor(0x00000000).build();
                    for (Light l : ringLights) {
                        rb.addLight(l, off);
                    }
                    session.requestLights(rb.build());
                    System.out.println("OK OFF");
                } else if ("SOLID".equals(cmd) && parts.length >= 2) {
                    int color = parseColor(parts[1]);
                    LightsRequest.Builder rb = new LightsRequest.Builder();
                    LightState st = new LightState.Builder().setColor(color).build();
                    for (Light l : ringLights) {
                        rb.addLight(l, st);
                    }
                    session.requestLights(rb.build());
                    System.out.println("OK SOLID " + parts[1]);
                } else if ("SEG".equals(cmd) && parts.length >= 2) {
                    LightsRequest.Builder rb = new LightsRequest.Builder();
                    for (int i = 0; i < ringLights.size(); i++) {
                        String hex = parts[1 + (i % (parts.length - 1))];
                        int color = parseColor(hex);
                        rb.addLight(ringLights.get(i), new LightState.Builder().setColor(color).build());
                    }
                    session.requestLights(rb.build());
                    System.out.println("OK SEG");
                } else {
                    System.out.println("ERR UNKNOWN " + line);
                }
                System.out.flush();
            }
        }
    }
}
