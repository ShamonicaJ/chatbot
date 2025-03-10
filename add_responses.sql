
CREATE TABLE IF NOT EXISTS book_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_patterns TEXT,    -- E.g., "Recommend a fantasy book"
    bot_responses TEXT,    -- E.g., "Try 'Mistborn' by Brandon Sanderson!"
    genre TEXT,            -- Optional: Filter by genre
    rating FLOAT           -- Optional: Include ratings
);

-- Insert sample book-themed responses
INSERT INTO book_responses (category, user_patterns, bot_responses)
VALUES 
    ('greeting', 'Hello,Hi,Hey', 'Hi! Want a book recommendation?'),
    ('genre', 'Recommend a fantasy book,Best sci-fi novels', 'Try "The Name of the Wind" or "Dune"!'),
    ('rating', 'Top-rated books,Best rated books', 'Check out "The Hobbit" or "1984"!');