/*
 * BotShield Store – main.js (Step 2)
 *
 * Basic UI interactions: mobile nav, product filters, tabs, quantity controls,
 * color swatches. No behavioural tracking yet – that comes in a later step.
 */

document.addEventListener("DOMContentLoaded", function () {

  // ----------------------------------------------------------------
  // Mobile nav toggle
  // ----------------------------------------------------------------
  const toggle = document.querySelector(".navbar__toggle");
  const links = document.querySelector(".navbar__links");

  if (toggle && links) {
    toggle.addEventListener("click", function () {
      links.classList.toggle("navbar__links--open");
    });
  }

  // ----------------------------------------------------------------
  // Product page – category filter buttons
  // ----------------------------------------------------------------
  const filterBtns = document.querySelectorAll(".filter-btn");
  const productCards = document.querySelectorAll(".products-grid__inner .product-card");

  filterBtns.forEach(function (btn) {
    btn.addEventListener("click", function () {
      filterBtns.forEach(function (b) { b.classList.remove("active"); });
      btn.classList.add("active");

      var category = btn.getAttribute("data-category");

      productCards.forEach(function (card) {
        if (category === "all" || card.getAttribute("data-category") === category) {
          card.style.display = "";
        } else {
          card.style.display = "none";
        }
      });
    });
  });

  // ----------------------------------------------------------------
  // Product detail – quantity controls
  // ----------------------------------------------------------------
  var qtyValue = document.getElementById("qty-value");
  var qtyMinus = document.getElementById("qty-minus");
  var qtyPlus = document.getElementById("qty-plus");

  if (qtyValue && qtyMinus && qtyPlus) {
    var qty = 1;

    qtyMinus.addEventListener("click", function () {
      if (qty > 1) {
        qty--;
        qtyValue.textContent = qty;
      }
    });

    qtyPlus.addEventListener("click", function () {
      if (qty < 10) {
        qty++;
        qtyValue.textContent = qty;
      }
    });
  }

  // ----------------------------------------------------------------
  // Product detail – color swatches
  // ----------------------------------------------------------------
  var swatches = document.querySelectorAll(".color-swatch");

  swatches.forEach(function (swatch) {
    swatch.addEventListener("click", function () {
      swatches.forEach(function (s) { s.classList.remove("active"); });
      swatch.classList.add("active");
    });
  });

  // ----------------------------------------------------------------
  // Product detail – tabs
  // ----------------------------------------------------------------
  var tabBtns = document.querySelectorAll(".product-detail__tab-btn");
  var tabPanels = document.querySelectorAll(".product-detail__tab-panel");

  tabBtns.forEach(function (btn) {
    btn.addEventListener("click", function () {
      var tabId = btn.getAttribute("data-tab");

      tabBtns.forEach(function (b) { b.classList.remove("active"); });
      tabPanels.forEach(function (p) { p.classList.remove("active"); });

      btn.classList.add("active");
      var panel = document.getElementById("tab-" + tabId);
      if (panel) panel.classList.add("active");
    });
  });

  // ----------------------------------------------------------------
  // Login form – prevent default (demo only)
  // ----------------------------------------------------------------
  var loginForm = document.getElementById("login-form");
  if (loginForm) {
    loginForm.addEventListener("submit", function (e) {
      e.preventDefault();
    });
  }

});

// ----------------------------------------------------------------
// BotShield Monitor (Step 5.1, hybrid FSM + ML scores)
// ----------------------------------------------------------------
function setMonitorText(id, text, unavailable) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.classList.toggle('monitor-val--na', !!unavailable);
}

function formatMonitorScore(value) {
  return typeof value === 'number' ? value.toFixed(2) : '—';
}

function updateBotShieldMonitor() {
  const monitorEl = document.getElementById('botshield-monitor');
  if (!monitorEl) return;

  fetch('/api/risk/current', { credentials: 'same-origin' })
    .then(res => res.json())
    .then(data => {
      // Last payload, for checking the panel against the backend
      window.__botshieldRisk = data;
      const signalsEl = document.getElementById('monitor-signals');

      if (!data.available) {
        setMonitorText('monitor-fsm', '—', true);
        setMonitorText('monitor-ml', '—', true);
        setMonitorText('monitor-hybrid', '—', true);
        setMonitorText('monitor-mode', data.message || 'Waiting for activity', true);
        if (signalsEl) signalsEl.innerHTML = '<li>No behavioral activity recorded yet</li>';
        return;
      }

      setMonitorText('monitor-fsm', formatMonitorScore(data.fsm.score));
      if (data.ml.active) {
        setMonitorText('monitor-ml', formatMonitorScore(data.ml.score));
      } else {
        // Never show a placeholder ML number when the model did not run
        setMonitorText('monitor-ml', 'N/A', true);
      }
      setMonitorText('monitor-hybrid', formatMonitorScore(data.hybrid.score));
      setMonitorText('monitor-risk', data.hybrid.risk_level);
      setMonitorText('monitor-mode', data.ml.active
        ? data.hybrid.scoring_mode
        : data.hybrid.scoring_mode + ' (ML ' + data.ml.status + ')');
      setMonitorText('monitor-state', data.fsm.state);
      setMonitorText('monitor-score', typeof data.automata.score === 'number'
        ? data.automata.state + ' · ' + data.automata.score.toFixed(2)
        : data.automata.state);

      if (signalsEl) {
        signalsEl.innerHTML = '';
        const reasons = data.reasons && data.reasons.length
          ? data.reasons
          : ['No suspicious behavior detected'];
        reasons.slice(0, 5).forEach(reason => {
          const li = document.createElement('li');
          li.textContent = reason;
          signalsEl.appendChild(li);
        });
      }
    })
    .catch(err => console.error("Monitor poll error:", err));
}

// Start polling if monitor is present
if (document.getElementById('botshield-monitor')) {
  updateBotShieldMonitor();
  setInterval(updateBotShieldMonitor, 1500);
}