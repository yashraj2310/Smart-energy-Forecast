// service-worker.js

const CACHE_NAME = "smart-energy-cache-v1";
const urlsToCache = [
  "./",
  "./index.html",
  "./style.css",
  "./app.js",
  "./manifest.json",
  "https://cdn.jsdelivr.net/npm/chart.js"
];

// Install event
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log("Caching app resources");
      return cache.addAll(urlsToCache);
    })
  );
});

// Fetch event (offline handling)
self.addEventListener("fetch", (event) => {
  const url = event.request.url;

  // 🚫 1. Don't touch API requests (POST/GET/WS)
  if (url.includes("/api/") || url.includes("/ws/")) {
    return; // Let the browser handle normally
  }

  // 🚫 2. Don't touch POST requests at all
  if (event.request.method !== "GET") {
    return;
  }

  // ✅ 3. Offline-first caching for frontend assets only
  event.respondWith(
    caches.match(event.request).then((response) => {
      return (
        response ||
        fetch(event.request).catch(() =>
          caches.match("./index.html")
        )
      );
    })
  );
});


// Activate event
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
});
