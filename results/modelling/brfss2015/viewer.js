/* Shared behaviour for local reports; no network requests and no HTML insertion. */
(() => {
  "use strict";
  const data = (id) => {
    const element = document.getElementById(id);
    return element ? JSON.parse(element.textContent) : [];
  };
  const comparisons = data("comparison-data");
  const select = document.getElementById("bmi");
  if (select) {
    for (const row of comparisons) {
      const option = document.createElement("option");
      option.value = row.state;
      option.textContent = row.state;
      select.append(option);
    }
    const show = () => {
      const row = comparisons.find((item) => item.state === select.value);
      if (!row) return;
      const percent = (value) => (100 * value).toFixed(2) + "%";
      document.getElementById("result").textContent =
        "Observed condition: " + percent(row.observational.distribution["1"]) +
        " · Hypothetical intervention: " + percent(row.interventional.distribution["1"]) +
        " · " + row.interventional.identification.status.replaceAll("_", " ");
    };
    select.addEventListener("change", show);
    show();
  }
  const samples = data("nlp-data");
  const picker = document.getElementById("nlp-example");
  if (picker) {
    for (const sample of samples) {
      const option = document.createElement("option");
      option.value = sample.id;
      option.textContent = sample.split + ": " + sample.id;
      picker.append(option);
    }
    const showSentence = () => {
      const sample = samples.find((item) => item.id === picker.value);
      if (!sample) return;
      document.getElementById("sentence").textContent = sample.sentence;
      document.getElementById("entities").textContent = "Mentions: " + sample.entities.join(" ↔ ");
      document.getElementById("nlp-predictions").textContent = Object.entries(sample.predictions)
        .map(([name, label]) => name + ": " + label).join(" · ");
    };
    picker.addEventListener("change", showSentence);
    showSentence();
  }
})();
