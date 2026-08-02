import { useEffect, useRef, useState } from "react";
import "./App.css";

interface FrameMessage {
  type: "frame";
  data: string;
  score: number;
  lives: number;
  gameOver: boolean;
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [score, setScore] = useState(0);
  const [lives, setLives] = useState(3);
  const [gameOver, setGameOver] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8765");
    socketRef.current = ws;

    ws.onmessage = (event) => {
      const msg: FrameMessage = JSON.parse(event.data);
      if (msg.type === "frame") {
        setScore(msg.score);
        setLives(msg.lives);
        setGameOver(msg.gameOver);

        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const img = new Image();
        img.onload = () => {
          canvas.width = img.width;
          canvas.height = img.height;
          ctx.drawImage(img, 0, 0);
        };
        img.src = "data:image/jpeg;base64," + msg.data;
      }
    };

    return () => ws.close();
  }, []);

  const handlePlayAgain = () => {
    socketRef.current?.send(JSON.stringify({ action: "reset" }));
  };

  const hearts = Array.from({ length: 3 }, (_, i) => (
    <span key={i} className="heart">{i < lives ? "❤️" : "🤍"}</span>
  ));

  return (
    <div className="game-wrapper">
      <canvas ref={canvasRef} className="game-canvas" />

      {!gameOver && (
        <div className="navbar">
          <div className="score">Score: {score}</div>
          <div className="hearts">{hearts}</div>
        </div>
      )}

      {gameOver && (
        <div className="game-over-backdrop">
          <div className="game-over-card">
            <h1>GAME OVER!</h1>
            <img src="/cat.gif" alt="sad cat" className="cat-gif" />
            <p className="final-score">Score: {score}</p>
            <button onClick={handlePlayAgain}>Play Again</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;