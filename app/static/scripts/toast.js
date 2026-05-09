(function () {
    function getContainer() {
        let c = document.getElementById('toast-container');
        if (!c) {
            c = document.createElement('div');
            c.id = 'toast-container';
            document.body.appendChild(c);
        }
        return c;
    }

    const icons = {
        error:   '✕',
        success: '✓',
        warning: '⚠',
        info:    'ℹ',
    };

    window.toast = function (message, type = 'info', duration = 3500) {
        const container = getContainer();

        const el = document.createElement('div');
        el.className = `toast toast-${type}`;
        el.innerHTML = `
            <span class="toast-icon">${icons[type] || icons.info}</span>
            <span class="toast-msg">${message}</span>
            <button class="toast-close" aria-label="Закрити">✕</button>
        `;

        container.appendChild(el);

        requestAnimationFrame(() => el.classList.add('toast-show'));

        el.querySelector('.toast-close').addEventListener('click', () => dismiss(el));

        const timer = setTimeout(() => dismiss(el), duration);

        el.addEventListener('mouseenter', () => clearTimeout(timer));
        el.addEventListener('mouseleave', () => {
            setTimeout(() => dismiss(el), 1000);
        });

        return el;
    };

    function dismiss(el) {
        el.classList.remove('toast-show');
        el.classList.add('toast-hide');
        el.addEventListener('transitionend', () => el.remove(), { once: true });
    }

    window.alert = function (message) {
        toast(String(message), 'warning');
    };

})();