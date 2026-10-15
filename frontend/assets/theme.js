(() => {
  let theme = window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
  try {
    const saved =
      localStorage.getItem("rag-theme") ||
      localStorage.getItem("ace-rag-theme");
    if (saved === "dark" || saved === "light") theme = saved;
  } catch {
    /* The theme still works when storage is disabled. */
  }
  document.documentElement.dataset.theme = theme;
})();
