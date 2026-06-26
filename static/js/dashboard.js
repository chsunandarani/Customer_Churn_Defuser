// Customer Churn Defuser - dashboard.js
// Handles sidebar toggle (mobile), Chart.js KPI charts, and small UX helpers.

document.addEventListener("DOMContentLoaded", function () {
  const toggleBtn = document.getElementById("sidebarToggle");
  const sidebar = document.getElementById("sidebar");

  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener("click", function () {
      sidebar.classList.toggle("show");
    });
  }

  // Animate KPI numbers counting up
  document.querySelectorAll(".kpi-value[data-target]").forEach(function (el) {
    const target = parseFloat(el.getAttribute("data-target")) || 0;
    const isDecimal = target % 1 !== 0;
    let current = 0;
    const steps = 40;
    const increment = target / steps;
    const suffix = el.getAttribute("data-suffix") || "";

    const interval = setInterval(function () {
      current += increment;
      if (current >= target) {
        current = target;
        clearInterval(interval);
      }
      el.textContent = (isDecimal ? current.toFixed(2) : Math.round(current)) + suffix;
    }, 20);
  });

  // Dashboard risk breakdown chart (Chart.js)
  const riskCanvas = document.getElementById("riskChart");
  if (riskCanvas && window.dashboardStats) {
    const stats = window.dashboardStats;
    new Chart(riskCanvas, {
      type: "doughnut",
      data: {
        labels: ["Safe", "Medium Risk", "High Risk"],
        datasets: [
          {
            data: [stats.safe, stats.medium_risk, stats.high_risk],
            backgroundColor: ["#2ECC71", "#F39C12", "#E74C3C"],
            borderWidth: 0,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: "bottom" },
          title: { display: true, text: "Customer Risk Breakdown" },
        },
      },
    });
  }

  const churnCanvas = document.getElementById("churnRateChart");
  if (churnCanvas && window.dashboardStats) {
    const stats = window.dashboardStats;
    new Chart(churnCanvas, {
      type: "bar",
      data: {
        labels: ["Active", "Churned"],
        datasets: [
          {
            label: "Customers",
            data: [stats.active_customers, stats.total_customers - stats.active_customers],
            backgroundColor: ["#2E86DE", "#EE5253"],
            borderRadius: 8,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false }, title: { display: true, text: "Active vs Churned" } },
        scales: { y: { beginAtZero: true } },
      },
    });
  }
});
