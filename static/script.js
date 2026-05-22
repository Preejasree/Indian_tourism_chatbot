document.addEventListener("DOMContentLoaded", () => {

const chatBox = document.querySelector('.chat-box');
const chatBtn = document.getElementById('chatBtn');
const closeBtn = document.getElementById('close-btn');
const sendBtn = document.getElementById('sendBtn');
const userInput = document.getElementById('userInput');
const chatBody = document.querySelector('.chat-box-body');

// ENTER KEY SUPPORT
userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") {
    sendBtn.click();
  }
});

// Open and close chat
chatBtn.addEventListener('click', () => {
  chatBox.style.display = 'flex';
  chatBtn.style.display = 'none';
});

closeBtn.addEventListener('click', () => {
  chatBox.style.display = 'none';
  chatBtn.style.display = 'block';
});

// SEND MESSAGE
sendBtn.addEventListener('click', async () => {

  const text = userInput.value.trim();

  if (!text) return;

  // USER MESSAGE
  const userMessage = document.createElement('div');
  userMessage.classList.add('message', 'user');
  userMessage.textContent = text;
  chatBody.appendChild(userMessage);

  userInput.value = '';

  // BOT PLACEHOLDER
  const botMessage = document.createElement('div');
  botMessage.classList.add('message', 'bot');
  botMessage.textContent = "Typing...";
  chatBody.appendChild(botMessage);

  chatBody.scrollTop = chatBody.scrollHeight;

  try {

    const res = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ message: text })
    });

    const data = await res.json();

    // FORMATTED RESPONSE
    //botMessage.innerHTML = data.response.replace(/\n/g, "<br>");
      botMessage.innerHTML = data.response
         .replace(/\n/g, "<br>")
         .replace(
         /(https?:\/\/[^\s]+)/g,
         '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
        );
      chatBody.scrollTop = chatBody.scrollHeight;

  } catch (error) {

    botMessage.textContent = "Error connecting to server.";
    console.error(error);
  }

});

});