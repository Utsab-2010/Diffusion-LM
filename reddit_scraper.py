import praw
from typing import List, Optional
import time

class RedditScraper:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """
        Initialize Reddit scraper with credentials.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string (e.g., "MyBot/1.0")
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    
    def get_posts(self, subreddit_name: str, limit: int = 100, 
                  sort_by: str = 'hot') -> List[dict]:
        """
        Fetch posts from a subreddit.
        
        Args:
            subreddit_name: Name of the subreddit
            limit: Number of posts to fetch
            sort_by: Sort method ('hot', 'new', 'top', 'rising')
        
        Returns:
            List of dictionaries containing post data
        """
        subreddit = self.reddit.subreddit(subreddit_name)
        posts = []
        
        if sort_by == 'hot':
            submissions = subreddit.hot(limit=limit)
        elif sort_by == 'new':
            submissions = subreddit.new(limit=limit)
        elif sort_by == 'top':
            submissions = subreddit.top(limit=limit)
        elif sort_by == 'rising':
            submissions = subreddit.rising(limit=limit)
        else:
            raise ValueError(f"Invalid sort_by: {sort_by}")
        
        for submission in submissions:
            posts.append({
                'id': submission.id,
                'title': submission.title,
                'selftext': submission.selftext,
                'score': submission.score,
                'num_comments': submission.num_comments,
                'created_utc': submission.created_utc
            })
        
        return posts
    
    def get_comments(self, subreddit_name: str, post_limit: int = 100,
                     comment_limit: Optional[int] = None) -> List[str]:
        """
        Fetch comments from posts in a subreddit.
        
        Args:
            subreddit_name: Name of the subreddit
            post_limit: Number of posts to fetch
            comment_limit: Max comments per post (None for all)
        
        Returns:
            List of comment texts
        """
        posts = self.get_posts(subreddit_name, limit=post_limit)
        all_comments = []
        
        for post in posts:
            submission = self.reddit.submission(id=post['id'])
            submission.comments.replace_more(limit=0)  # Remove "More Comments" objects
            
            comments = submission.comments.list()
            if comment_limit:
                comments = comments[:comment_limit]
            
            for comment in comments:
                if hasattr(comment, 'body'):
                    all_comments.append(comment.body)
        
        return all_comments
    
    def get_all_text(self, subreddit_name: str, post_limit: int = 100,
                     comment_limit: Optional[int] = None, 
                     include_titles: bool = True) -> str:
        """
        Get all text from a subreddit (posts and comments combined).
        
        Args:
            subreddit_name: Name of the subreddit
            post_limit: Number of posts to fetch
            comment_limit: Max comments per post
            include_titles: Whether to include post titles
        
        Returns:
            Combined text from all posts and comments
        """
        print(f"Fetching data from r/{subreddit_name}...")
        posts = self.get_posts(subreddit_name, limit=post_limit)
        
        all_text = []
        
        # Add post titles and selftext
        for post in posts:
            if include_titles and post['title']:
                all_text.append(post['title'])
            if post['selftext']:
                all_text.append(post['selftext'])
        
        print(f"Fetched {len(posts)} posts")
        
        # Add comments
        print("Fetching comments...")
        comments = self.get_comments(subreddit_name, post_limit, comment_limit)
        all_text.extend(comments)
        
        print(f"Fetched {len(comments)} comments")
        print(f"Total text entries: {len(all_text)}")
        
        return "\n\n".join(all_text)
    
    def save_to_file(self, text: str, filename: str):
        """Save text to a file."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Saved to {filename}")


# Example usage:
if __name__ == "__main__":
    # You need to create a Reddit app at https://www.reddit.com/prefs/apps
    # to get client_id and client_secret
    
    scraper = RedditScraper(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET",
        user_agent="MyRedditScraper/1.0"
    )
    
    # Get all text from a subreddit
    text = scraper.get_all_text("MachineLearning", post_limit=50)
    
    # Save to file
    scraper.save_to_file(text, "reddit_data.txt")
