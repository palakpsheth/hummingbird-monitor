
// Global lazy loading using IntersectionObserver
document.addEventListener('DOMContentLoaded', function () {
    const lazyImages = document.querySelectorAll('img.lazy-thumb');

    if (lazyImages.length > 0 && 'IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries, obs) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    if (img.dataset.src) {
                        img.src = img.dataset.src;
                        img.classList.remove('lazy-thumb');

                        // Optional: add a class when loaded to allow for CSS transitions
                        img.onload = () => img.classList.add('loaded');
                        // In case it's already cached
                        if (img.complete) img.classList.add('loaded');

                        obs.unobserve(img);
                    }
                }
            });
        }, {
            rootMargin: '200px' // Start loading before they come into view
        });

        lazyImages.forEach(img => observer.observe(img));
    } else {
        // Fallback for no IO support or just to be safe: load all immediately
        lazyImages.forEach(img => {
            if (img.dataset.src) {
                img.src = img.dataset.src;
                img.classList.remove('lazy-thumb');
            }
        });
    }
});
