package com.osmo.face;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.view.Window;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

public class MainActivity extends Activity {
    private static final String DEFAULT_URL = "http://127.0.0.1:8765/face";
    private static final int OBSIDIAN = 0xFF07080B;

    private WebView webView;
    private String currentUrl = DEFAULT_URL;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final Runnable retryLoad = new Runnable() {
        @Override
        public void run() {
            if (webView != null) {
                webView.loadUrl(currentUrl);
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        setShowWhenLocked(true);
        setTurnScreenOn(true);

        Window win = getWindow();
        win.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        win.setDecorFitsSystemWindows(false);

        String extraUrl = getIntent() != null ? getIntent().getStringExtra("url") : null;
        if (extraUrl != null && !extraUrl.isEmpty()) {
            currentUrl = extraUrl;
        }

        webView = new WebView(this);
        webView.setBackgroundColor(OBSIDIAN);
        webView.setOverScrollMode(View.OVER_SCROLL_NEVER);
        webView.setVerticalScrollBarEnabled(false);
        webView.setHorizontalScrollBarEnabled(false);

        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setCacheMode(WebSettings.LOAD_NO_CACHE);
        settings.setMediaPlaybackRequiresUserGesture(false);
        settings.setUseWideViewPort(true);
        settings.setLoadWithOverviewMode(true);
        settings.setSupportZoom(false);
        settings.setBuiltInZoomControls(false);

        webView.setWebChromeClient(new WebChromeClient());
        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
                if (request != null && request.isForMainFrame()) {
                    String waitingHtml =
                        "<html><body style='margin:0;background:#07080b;color:#e8edf5;" +
                        "font-family:monospace;display:flex;align-items:center;justify-content:center;" +
                        "height:100vh;letter-spacing:0.16em;font-size:12px;'>" +
                        "<div style='text-align:center;opacity:0.75;'>" +
                        "<div style='margin-bottom:8px;color:#60a5fa;'>OSMO // OLED KIOSK</div>" +
                        "<div>CONNECTING TO STUDIO (127.0.0.1:8765)...</div>" +
                        "</div></body></html>";
                    view.loadDataWithBaseURL(null, waitingHtml, "text/html", "utf-8", null);
                    handler.removeCallbacks(retryLoad);
                    handler.postDelayed(retryLoad, 1500);
                }
            }
        });

        webView.clearCache(true);
        setContentView(webView);
        applyImmersiveFullscreen();
        webView.loadUrl(currentUrl);
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        String extraUrl = intent != null ? intent.getStringExtra("url") : null;
        if (extraUrl != null && !extraUrl.isEmpty()) {
            currentUrl = extraUrl;
        }
        applyImmersiveFullscreen();
        if (webView != null) {
            handler.removeCallbacks(retryLoad);
            webView.clearCache(true);
            webView.loadUrl(currentUrl);
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        applyImmersiveFullscreen();
        if (webView != null) {
            webView.onResume();
        }
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) {
            applyImmersiveFullscreen();
        }
    }

    @Override
    @SuppressWarnings("deprecation")
    public void onBackPressed() {
        // Kiosk guard: ignore accidental back gestures during live demos
    }

    @Override
    protected void onDestroy() {
        handler.removeCallbacks(retryLoad);
        if (webView != null) {
            webView.destroy();
            webView = null;
        }
        super.onDestroy();
    }

    private void applyImmersiveFullscreen() {
        WindowInsetsController controller = getWindow().getInsetsController();
        if (controller != null) {
            controller.hide(WindowInsets.Type.statusBars() | WindowInsets.Type.navigationBars());
            controller.setSystemBarsBehavior(WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
        }
    }
}
