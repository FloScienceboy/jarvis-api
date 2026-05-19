// JARVIS Service Worker — Offline Cache + Push
const CACHE = 'jarvis-v1';
const ASSETS = ['/', '/index.html', '/manifest.json'];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  // API calls: network first, no cache
  if (url.pathname.startsWith('/api') || url.pathname.startsWith('/ws')) return;
  // Static: cache first
  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request))
  );
});

// Push notifications (for WhatsApp bot alerts)
self.addEventListener('push', e => {
  const data = e.data ? e.data.json() : { title: 'JARVIS', body: 'Neue Benachrichtigung' };
  e.waitUntil(
    self.registration.showNotification(data.title || 'JARVIS', {
      body: data.body || '',
      icon: '/icons/icon-192.png',
      badge: '/icons/icon-192.png',
      vibrate: [200, 100, 200],
      data: { url: data.url || '/' },
      actions: [
        { action: 'open', title: 'Öffnen' },
        { action: 'dismiss', title: 'Schließen' },
      ],
    })
  );
});

self.addEventListener('notificationclick', e => {
  e.notification.close();
  if (e.action === 'dismiss') return;
  e.waitUntil(clients.openWindow(e.notification.data.url || '/'));
});
