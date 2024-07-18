import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
import os

from config import ALLOWED_DOMAINS, START_URLS, DEPTH_LIMIT, OUTPUT_FILE

class NvidiaCudaSpider(CrawlSpider):
    name = 'nvidia_cuda'
    allowed_domains = ALLOWED_DOMAINS
    start_urls = START_URLS

    rules = (
        Rule(LinkExtractor(allow=r'/cuda/'), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        yield {
            'url': response.url,
            'title': response.css('title::text').get(),
            'content': ' '.join(response.css('p::text').getall()),
        }

def run_spider():
    os.makedirs('output', exist_ok=True)

    settings = get_project_settings()
    settings.update({
        'FEED_FORMAT': 'json',
        'FEED_URI': OUTPUT_FILE,
        'DEPTH_LIMIT': DEPTH_LIMIT,
        'LOG_LEVEL': 'INFO'
    })

    process = CrawlerProcess(settings)
    process.crawl(NvidiaCudaSpider)
    process.start()

if __name__ == "__main__":
    run_spider()
    print("Crawling completed. Check the 'output' directory for the cuda_docs.json file.")