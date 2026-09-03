window.HELP_IMPROVE_VIDEOJS = false;

function fallbackCopy(text) {
  const textArea = document.createElement('textarea');
  textArea.value = text;
  textArea.setAttribute('readonly', '');
  textArea.style.position = 'fixed';
  textArea.style.opacity = '0';
  document.body.appendChild(textArea);
  textArea.select();
  const copied = document.execCommand('copy');
  document.body.removeChild(textArea);
  return copied;
}

async function copyBibTeX() {
  const bibtexElement = document.getElementById('bibtex-code');
  const button = document.querySelector('.copy-bibtex-btn');
  const copyText = button?.querySelector('.copy-text');

  if (!bibtexElement || !button || !copyText) return;

  let copied = false;
  try {
    await navigator.clipboard.writeText(bibtexElement.textContent);
    copied = true;
  } catch (error) {
    copied = fallbackCopy(bibtexElement.textContent);
  }

  copyText.textContent = copied ? 'Copied!' : 'Copy failed';
  button.classList.toggle('copied', copied);

  window.setTimeout(() => {
    button.classList.remove('copied');
    copyText.textContent = 'Copy';
  }, 2000);
}

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function setupScrollButton() {
  const scrollButton = document.querySelector('.scroll-to-top');
  if (!scrollButton) return;

  const update = () => {
    scrollButton.classList.toggle('visible', window.scrollY > 480);
  };

  window.addEventListener('scroll', update, { passive: true });
  update();
}

function setupVideoPlayback() {
  const videos = [...document.querySelectorAll('.results-carousel video')];
  if (!videos.length) return;

  videos.forEach((video) => {
    video.addEventListener('play', () => {
      videos.forEach((otherVideo) => {
        if (otherVideo !== video) otherVideo.pause();
      });
    });
  });

  if (!('IntersectionObserver' in window)) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) entry.target.pause();
      });
    },
    { threshold: 0.2 },
  );

  videos.forEach((video) => observer.observe(video));
}

function setupCarousels() {
  if (!window.bulmaCarousel) return;

  window.bulmaCarousel.attach('.carousel', {
    slidesToScroll: 1,
    slidesToShow: 1,
    loop: true,
    infinite: true,
    autoplay: false,
  });
}

document.addEventListener('DOMContentLoaded', () => {
  setupCarousels();
  setupScrollButton();
  setupVideoPlayback();
});
