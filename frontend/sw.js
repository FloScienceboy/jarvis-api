// JARVIS Service Worker — Auto-Update + Offline Cache
const CACHE_KEY = 'jarvis-v2';
const STATIC_ASSETS = ['/manifest.json', '/icons/icon-192.png', '/icons/icon-512.png'];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE_KEY).then(c => c.addAll(STATIC_ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE_KEY).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (url.pathname.startsWith('/api') || url.pathname.startsWith('/ws') ||
      url.pathname.startsWith('/__version__')) return;
  if (url.pathname === '/' || url.pathname === '/index.html') {
    e.respondWith(
      fetch(e.request).then(res => {
        caches.open(CACHE_KEY).then(c => c.put(e.request, res.clone()));
        return res;
      }).catch(() => caches.match('/index.html'))
    );
    return;
  }
  e.respondWith(caches.match(e.request).then(cached => cached || fetch(e.request)));
});

let currentVersion = null;
async function checkForUpdate() {
  try {
    const res = await fetch('/__version__?t=' + Date.now());
    const data = await res.json();
    if (!currentVersion) { currentVersion = data.version; return; }
    if (data.version !== currentVersion) {
      currentVersion = data.version;
      const clients = await self.clients.matchAll({ type: 'window' });
      clients.forEach(c => c.postMessage({ type: 'NEW_VERSION' }));
    }
  } catch (_) {}
}

self.addEventListener('activate', () => {
  setTimeout(checkForUpdate, 10000);
  setInterval(checkForUpdate, 300000);
});

self.addEventListener('push', e => {
  const data = e.data ? e.data.json() : { title: 'JARVIS', body: 'Neue Benachrichtigung' };
  e.waitUntil(self.registration.showNotification(data.title || 'JARVIS', {
    body: data.body || '', icon: '/icons/icon-192.png', badge: '/icons/icon-192.png',
    vibrate: [200, 100, 200], data: { url: data.url || '/' },
    actions: [{ action: 'open', title: 'Oeffnen' }, { action: 'dismiss', title: 'Schliessen' }],
  }));
});

self.addEventListener('notificationclick', e => {
  e.notification.close();
  if (e.action === 'dismiss') return;
  e.waitUntil(clients.openWindow(e.notification.data.url || '/'));
});
