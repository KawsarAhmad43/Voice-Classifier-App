let mediaRecorder;
let audioBlob;

document.getElementById('recordButton').addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        const chunks = [];
        mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
        mediaRecorder.onstop = () => {
            audioBlob = new Blob(chunks, { type: 'audio/wav' });
            document.getElementById('predictButton').classList.remove('hidden');
            document.getElementById('status').textContent = 'Recording stopped. Click Predict.';
        };
        mediaRecorder.start();
        document.getElementById('recordButton').classList.add('hidden');
        document.getElementById('status').textContent = 'Recording... Speak now!';
        setTimeout(() => {
            if (mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                document.getElementById('status').textContent = 'Recording stopped (5s limit). Click Predict.';
            }
        }, 5000);
    } catch (err) {
        document.getElementById('status').textContent = 'Error accessing microphone: ' + err.message;
    }
});

document.getElementById('predictButton').addEventListener('click', () => {
    document.getElementById('predictButton').classList.add('hidden');
    document.getElementById('loader').classList.remove('hidden');
    document.getElementById('status').textContent = '';
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');
    fetch('/predict', {
        method: 'POST',
        body: formData
    }).then(response => response.text())
      .then(html => {
          document.body.innerHTML = html;
      }).catch(err => {
          document.getElementById('loader').classList.add('hidden');
          document.getElementById('status').textContent = 'Error sending audio: ' + err.message;
      });
});