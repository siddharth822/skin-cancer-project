(() => {
  const root = document.documentElement;
  try { root.dataset.theme = localStorage.getItem('skinsight-theme') || 'dark'; } catch (_) {}
  const toggle = document.getElementById('themeToggle');
  const updateLabel = () => toggle?.setAttribute('aria-label', `Switch to ${root.dataset.theme === 'dark' ? 'light' : 'dark'} theme`);
  updateLabel();
  toggle?.addEventListener('click', () => {
    root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    try { localStorage.setItem('skinsight-theme', root.dataset.theme); } catch (_) {}
    updateLabel();
  });
  const nav = document.getElementById('mainNav');
  document.getElementById('navToggle')?.addEventListener('click', event => {
    const open = nav.classList.toggle('open');
    event.currentTarget.setAttribute('aria-expanded', String(open));
  });
})();
