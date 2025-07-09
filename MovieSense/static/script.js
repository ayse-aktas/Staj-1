function autocompleteSearch() {
    const input = document.getElementById("searchBox");
    const query = input.value;
    const suggestionBox = document.getElementById("suggestions");

    if (query.length < 2) {
        suggestionBox.innerHTML = "";
        return;
    }

    fetch(`/autocomplete?q=${query}`)
        .then(response => response.json())
        .then(data => {
            suggestionBox.innerHTML = "";
            data.forEach(title => {
                const item = document.createElement("div");
                item.textContent = title;
                item.onclick = () => {
                    input.value = title;
                    suggestionBox.innerHTML = "";
                    input.form.submit();  // isteğe bağlı: otomatik gönder
                };
                suggestionBox.appendChild(item);
            });
        });
}
