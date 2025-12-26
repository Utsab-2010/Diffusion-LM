
import praw
import re


class RedditScraper:
    """Scrape posts and comments from Reddit subreddits."""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    
    def clean_text(self, text: str) -> str:
        """Remove URLs, GIF indicators, and other unwanted patterns."""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove markdown links [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove GIF indicators
        text = re.sub(r'!\[gif\]\([^\)]+\)', '', text)
        text = re.sub(r'https://giphy\.com/\S+', '', text)
        
        # Remove image indicators
        text = re.sub(r'!\[img\]\([^\)]+\)', '', text)
        
        # Clean up multiple newlines and whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()
    
    def scrape_and_save(self, subreddit_name: str, n_posts: int, output_file: str):
        """Scrape top posts and save to text file."""
        subreddit = self.reddit.subreddit(subreddit_name)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for submission in subreddit.new(limit=n_posts):
                title = self.clean_text(submission.title)
                f.write(f"Title: {title}\n")
                
                if submission.selftext:
                    cleaned_text = self.clean_text(submission.selftext)
                    if cleaned_text:
                        f.write(f"\n{cleaned_text}\n")
                
                # submission.comments.replace_more(limit=0)
                # for comment in submission.comments.list():
                #     if hasattr(comment, 'body'):
                #         cleaned_comment = self.clean_text(comment.body)
                #         if cleaned_comment:
                #             f.write(f"\n{cleaned_comment}\n")
                
                # f.write("\n" + "="*80 + "\n\n")
        
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    scraper = RedditScraper(
        client_id="732OOkEmzLdwSa5VjUxAzg",
        client_secret="HkoXIFJ5JXSvuGZJHHvBYbIUlB0sog",
        user_agent="content_bot"
    )
    
    scraper.scrape_and_save("nosleep", n_posts=200, output_file="nosleep_reddit.txt")