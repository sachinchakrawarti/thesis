/* ---------- Dark mode ---------- */
const themeBtn = document.getElementById('themeBtn');
const saved = localStorage.getItem('theme');
if (saved === 'dark') document.documentElement.setAttribute('data-theme', 'dark');

themeBtn.addEventListener('click', () => {
  const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
});

/* ---------- Print ---------- */
document.getElementById('printBtn').addEventListener('click', () => window.print());

/* ---------- Warn if content overflows a page ---------- */
window.addEventListener('load', () => {
  document.querySelectorAll('.page').forEach(p => {
    // scrollHeight > clientHeight means content is clipped
    if (p.scrollHeight > p.clientHeight + 2) {
      console.warn(
        `⚠️ Page ${p.dataset.page} content overflows A4 height by ${p.scrollHeight - p.clientHeight}px — content is being clipped. Trim text or reduce font size.`
      );
    }
  });
});