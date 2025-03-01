document
  .getElementById("ladderForm")
  .addEventListener("submit", function (event) {
    event.preventDefault();

    const startWord = document.getElementById("startWord").value.toLowerCase();
    const endWord = document.getElementById("endWord").value.toLowerCase();
    const algorithm = document.getElementById("algorithm").value;

    fetch("/generate_ladder", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        start_word: startWord,
        end_word: endWord,
        algorithm: algorithm,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        const ladder = data.ladder;
        const ladderList = document.getElementById("ladderList");
        ladderList.innerHTML = "";

        if (ladder.length > 0) {
          ladder.forEach((word) => {
            const listItem = document.createElement("li");
            listItem.textContent = word;
            ladderList.appendChild(listItem);
          });
        } else {
          ladderList.textContent = "No path found.";
        }
      })
      .catch((error) => console.error("Error:", error));
  });

document
  .getElementById("randomWordsButton")
  .addEventListener("click", function () {
    fetch("/random_words")
      .then((response) => response.json())
      .then((data) => {
        document.getElementById("startWord").value = data.start_word;
        document.getElementById("endWord").value = data.end_word;
      });
  });
