
import React, { useState } from 'react';
import axios from 'axios';
import { Box, TextField, Button, List, ListItem, Paper } from '@mui/material';

function App() {
  const [messages, setMessages] = useState([{
    text: "📚 Hi! I'm Book Buddy. Ask me for recommendations!",
    isUser: false
  }]);
  const [input, setInput] = useState('');

  const handleSend = async () => {
    if (!input.trim()) return;

    // Add user message
    setMessages(prev => [...prev, { text: input, isUser: true }]);
    
    try {
      const response = await axios.post('/.netlify/functions/chatbot', {
        message: input
      });

      setMessages(prev => [...prev, { 
        text: response.data.response, 
        isUser: false 
      }]);
    } catch (error) {
      console.error("API Error:", error);
      setMessages(prev => [...prev, { 
        text: "⚠️ Oops! The chatbot is unavailable.", 
        isUser: false 
      }]);
    }

    setInput('');
  };

  return (
    <Box sx={{ maxWidth: 800, margin: '2rem auto', padding: 2 }}>
      <Paper elevation={3} sx={{ padding: 2 }}>
        <List sx={{ height: '60vh', overflow: 'auto', mb: 2 }}>
          {messages.map((msg, i) => (
            <ListItem key={i} sx={{ 
              justifyContent: msg.isUser ? 'flex-end' : 'flex-start',
              alignItems: 'flex-start'
            }}>
              <Box sx={{
                bgcolor: msg.isUser ? 'primary.main' : 'grey.100',
                color: msg.isUser ? 'white' : 'text.primary',
                p: 2,
                borderRadius: 2,
                maxWidth: '70%',
                wordBreak: 'break-word'
              }}>
                {msg.text}
              </Box>
            </ListItem>
          ))}
        </List>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <TextField
            fullWidth
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Type your book request..."
            variant="outlined"
          />
          <Button 
            variant="contained" 
            onClick={handleSend}
            sx={{ height: '56px' }}
          >
            Send
          </Button>
        </Box>
      </Paper>
    </Box>
  );
}

export default App;
