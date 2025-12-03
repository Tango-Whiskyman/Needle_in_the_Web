import base64
from curses import raw
from firecrawl import FirecrawlApp, ScrapeOptions
import requests
import sys
import os
import time
import re
import browsergym.core
import gymnasium
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str
from browsergym.core.action.functions import goto, page, get_elem_by_bid, demo_mode, tab_focus
from browsergym.core.observation import _pre_extract, extract_dom_snapshot
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from io import StringIO
import lxml
import lxml.etree
import json
from pathlib import Path
import glob
from typing import Literal, Union
from NiW.constants import COOKIE_STRING_FT, COOKIE_STRING_WP
# from constants import COOKIE_STRING_FT, COOKIE_STRING_WP

from sympy import content

wd = Path(__file__).parent.resolve()

firecrawl_client = FirecrawlApp(api_key = None)

cache = dict()

def download_file(url: str, file_name: str):
    """
    Download a file from a given URL and save it to the 'temp' subdirectory with the specified file name.

    Args:
        url (str): the URL of the file to download.
        file_name (str): the name of the file to save.
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.getcwd() + "/temp/" + file_name, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully: {file_name}")
    else:
        print(f"Failed to download file: {url} (Status code: {response.status_code})")
    
def get_page_content(url: str, timeout: int = 300000, wait_for: int = 0):
    """
    Scrape the content of a given URL using Firecrawl.
    The result will be in Markdown format with base64 images removed.

    Args:
        url (str): the URL to scrape.
    """
    global cache
    if len(cache.keys()) == 0:
        with open("misc/webpage_cache/cache.json", "r") as f:
            cache = json.loads(f.read())
    if cache.get(url) is not None:
        return cache[url]
    scrape_result = None
    cookie_string = ""
    for i in range(10):
        try:
            if url.find("www.ft.com") != -1:
                scrape_result = firecrawl_client.scrape_url(url, formats=['markdown'], remove_base64_images=True, timeout=timeout, wait_for=wait_for, headers={"Cookie": COOKIE_STRING_FT})
            elif url.find("www.washingtonpost.com") != -1:
                scrape_result = firecrawl_client.scrape_url(url, formats=['markdown'], remove_base64_images=True, timeout=timeout, wait_for=wait_for, headers={"Cookie": COOKIE_STRING_WP})
            elif url.find("docs.ufpr.br") != -1:
                scrape_result = firecrawl_client.scrape_url(url, formats=['markdown'], remove_base64_images=True, timeout=timeout, wait_for=wait_for, parse_pdf=False)
                # parse the result from base64 encoding to text
                scrape_result.markdown = base64.b64decode(scrape_result.markdown).decode('utf-8')
            else:
                scrape_result = firecrawl_client.scrape_url(url, formats=['markdown'], remove_base64_images=True, timeout=timeout, wait_for=wait_for)
            cache[url] = scrape_result.markdown
            with open("misc/webpage_cache/cache.json", "w") as f:
                f.write(json.dumps(cache, indent=4))
            break
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            continue
    if scrape_result is None:
        print(f"Failed to scrape {url} after 10 attempts.")
        return ""		    
    return scrape_result.markdown

class QueryContextPage:
    
    def __init__(self, title: str, url: str, content: str):
        """
        Initialize a QueryContextPage object.

        Args:
            title (str): the title of the page.
            url (str): the URL of the page.
            content (str): the content of the page.
        """
        self.title = title
        self.url = url
        self.content = content
    
    def __str__(self):
        return f"Title: {self.title}\n\nURL: {self.url}\n\nContent: {self.content}"
    
    def json(self):
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content
        }
    
    title: str
    url: str
    content: str

def remove_links_from_markdown(content: str) -> str:
    while content.find("](http") != -1:
        index = content.find("](http")
        start_index = content.rfind("[", 0, index)
        end_index = content.find(")", index)
        content = content.replace(content[start_index:end_index + 1], content[start_index + 1:index])

    return content
    
def load_cookies_from_json(json_path):
    with open(json_path, 'r') as f:
        cookie_text = f.read()
        cookie_text = cookie_text.replace("no_restriction", "None")
        cookie_text = cookie_text.replace("lax", "Lax")
        cookie_text = cookie_text.replace("strict", "Strict")
        cookie_text = cookie_text.replace("\"sameSite\": null", "\"sameSite\": \"None\"")
        cookies = json.loads(cookie_text)
    return cookies


def convert_cookies_to_python():
    all_cookies = []
    # cookie_files = [
    #     "orcid.org.cookies.json",
    #     "www.researchgate.net.cookies.json",
    #     "github.com.cookies.json",
    #     "www.youtube.com.cookies.json",
    #     "www.ncbi.nlm.nih.gov.cookies.json",
    #     "archive.org.cookies.json", 
    #     "nature.com.cookies.json"
    # ]
    json_dir = wd / "cookie_json"
    cookie_files = glob.glob(str(json_dir / "*.json"))
    
    for cookie_file in cookie_files:
        json_path = wd / "cookie_json" / cookie_file
        cookies = load_cookies_from_json(json_path)
        all_cookies.extend(cookies)
    
    # 生成Python格式的cookies文件
    output_path = wd / "cookies_data.py"
    output_str = "COOKIES_LIST = [\n"
    for cookie in all_cookies:
        output_str += f"    {repr(cookie)},\n"
    output_str += "]\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_str)
    return output_str

def get_arxiv_abstract_and_introduction(experiment_id, subjects: list[str], limit: int = 3):
    def extract_abstract_and_introduction(paper_content: str):
        paper_content = paper_content.replace("Report issue for preceding element\n", "")
        lines = paper_content.split("\n")
        introduction_section_title = None
        hashtags = ""
        for line in lines:
            if line.startswith("#") and "introduction" in line.lower():
                introduction_section_title = line.strip()
                hashtags = line[:line.find(" ")]
                break
        if introduction_section_title is None:
            return None
        introduction_start_index = paper_content.find(introduction_section_title)
        introduction_end_index = paper_content.find(f"\n{hashtags} ", introduction_start_index + 1)
        paper_content = paper_content[:introduction_end_index].strip()
        index = paper_content.find("](http")
        while index != -1:
            start_index = paper_content.rfind("[", 0, index)
            end_index = paper_content.find(")", index)
            should_delete = True
            for char in paper_content[start_index + 1:index]:
                if not char.isdigit():
                    should_delete = False
                    break
            if should_delete:
                paper_content = paper_content.replace(paper_content[start_index:end_index + 1], paper_content[start_index + 1:index])
                index = paper_content.find("](http", start_index)
            else:
                index = paper_content.find("](http", end_index)
        return paper_content
    
    def clean_content(content: str):
        if content.find("[License:") != -1:
            content = content[content.find("[License:"):]
        index = content.find("\\[")
        while index != -1:
            end_index = content.find("\\]", index)
            content = content[:index] + content[end_index + 2:]
            index = content.find("\\[", index)
        index = content.find("](http")
        while index != -1:
            start_index = content.rfind("[", 0, index)
            end_index = content.find(")", index)
            content = content.replace(content[start_index:end_index], content[start_index + 1:index])
            index = content.find("](http")
        index = content.find("](mailto")
        while index != -1:
            start_index = content.rfind("[", 0, index)
            end_index = content.find(")", index)
            content = content.replace(content[start_index:end_index], "")
            index = content.find("](mailto")
        return content
        
    paper_list = []
    for subject in subjects:
        raw_arxiv_results = get_page_content(f"https://arxiv.org/list/{subject}/recent", wait_for=5000)
        tmp_paper_list = []
        for i in range(1, 1000):
            index = raw_arxiv_results.find(f"[{i}\\]")
            if index == -1 or len(tmp_paper_list) >= limit:
                break
            html_index = raw_arxiv_results.find("[html]", index)
            html_link = raw_arxiv_results[html_index + 7:raw_arxiv_results.find(" ", html_index)]
            # maybe directly converting the papers to md using firecrawl would be better...?
            paper_content = get_page_content(html_link, timeout=300000)
            abstract_and_introduction = extract_abstract_and_introduction(paper_content)
            abstract_and_introduction = clean_content(abstract_and_introduction)
            if abstract_and_introduction is not None:
                tmp_paper_list.append(QueryContextPage(
                    title=raw_arxiv_results[raw_arxiv_results.find("Title:", index) + 7:raw_arxiv_results.find("\n", raw_arxiv_results.find("Title:", index) + 8)].strip(),
                    url=html_link,
                    content=abstract_and_introduction
                ))
        paper_list.extend(tmp_paper_list)

    with open(f"experiments/{str(experiment_id)}/web_contents/{time.strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        f.write(json.dumps([page.json() for page in paper_list], indent=4))
    return paper_list

def get_arxiv_papers(subject: str, limit: int = 3):
    """
    Get the newest arXiv papers for a given subject.

    Args:
        subject (str): the subject to search for. It should be a valid arXiv subject like 'cs.AI'.
        limit (int): the maximum number of papers to return. Maximum is 50.
    """
    raw_arxiv_results = get_page_content(f"https://arxiv.org/list/{subject}/recent", wait_for=5000)
    paper_list = []
    for i in range(1, limit + 1):
        index = raw_arxiv_results.find(f"[{i}\\]")
        html_index = raw_arxiv_results.find("[html]", index)
        html_link = raw_arxiv_results[html_index + 7:raw_arxiv_results.find(" ", html_index)]
        # maybe directly converting the papers to md using firecrawl would be better...?
        paper_content = get_page_content(html_link, timeout=300000)
        paper_list.append(QueryContextPage(
            title = raw_arxiv_results[raw_arxiv_results.find("Title:", index) + 7:raw_arxiv_results.find("\n", raw_arxiv_results.find("Title:", index) + 8)].strip(),
            url = html_link,
            content = paper_content))
        # filename = f"{subject.replace('.', '_')}_{i}.pdf"
        # download_file(pdf_link, filename)
        # paper_list.append({
        #     "title": raw_arxiv_results[raw_arxiv_results.find("Title:", index) + 7:raw_arxiv_results.find("\n", raw_arxiv_results.find("Title:", index) + 8)].strip(),
        #     "pdf_link": pdf_link,
        #     "filename": filename
        # })
    return paper_list

def get_the_guardian_news(topic: str = "sport", limit: int = 3):
    raw_the_guardian_results = get_page_content(f"https://www.theguardian.com/uk/{topic}")
    news_list = []
    index = 0
    for i in range(1, limit + 1):
        index = raw_the_guardian_results.find(f"[**", index)
        if index == -1:
            break
        title_start = index + 3
        title_end = raw_the_guardian_results.find("**]", title_start)
        title = raw_the_guardian_results[title_start:title_end].strip()
        url_start = raw_the_guardian_results.find("**](", title_start) + 4
        url_end = raw_the_guardian_results.find(")", url_start)
        url = raw_the_guardian_results[url_start:url_end].strip()
        content = get_page_content(url)
        def clean_content(content: str):
            while content.find("](https://i.guim.co.uk") != -1:
                index = content.find("](https://i.guim.co.uk")
                start_index = content.rfind("![", 0, index)
                end_index = content.find(")", index)
                content = content.replace(content[start_index:end_index + 1], "")
            while content.find("Share](") != -1:
                index = content.find("Share](")
                start_index = content.rfind("[", 0, index)
                end_index = content.find(")", index)
                content = content.replace(content[start_index:end_index + 1], "")
            while content.find("[Reuse this content](") != -1:
                start_index = content.find("[Reuse this content](")
                end_index = content.find(")", start_index)
                content = content.replace(content[start_index:end_index + 1], "")
            content = content.replace("Explore more on these topics\n", "")
            while content.find("\n- [") != -1:
                start_index = content.find("\n- [")
                end_index = content.find(")", start_index)
                content = content.replace(content[start_index:end_index + 1], "")
            while content.find("Read more]") != -1:
                index = content.find("Read more]")
                start_index = content.rfind("[", 0, index)
                end_index = content.find(")", index)
                content = content.replace(content[start_index:end_index + 1], "")
            while content.find("[View image in fullscreen]") != -1:
                start_index = content.find("[View image in fullscreen]")
                end_index = content.find(")", start_index)
                content = content.replace(content[start_index:end_index + 1], "")
            while content.find("# Most viewed") != -1:
                index = content.find("# Most viewed")
                start_index = content.rfind("\n", 0, index)
                content = content[:start_index]
            return content.strip()
        content = clean_content(content)
        news_list.append(QueryContextPage(
            title = title,
            url = url,
            content = content
        ))
        index = title_end
    return news_list

def get_flight_information(from_city: str, from_airport: str, to_city: str, to_airport: str, date: str, adults: int = 1, children: int = 0, infants: int = 0, ):
    request_url = f"https://www.trip.com/flights/showfarefirst?dcity={from_city.lower()}&acity={to_city.lower()}&ddate={date}&dairport={from_airport.lower()}&aairport={to_airport.lower()}&triptype=ow&class=y&lowpricesource=searchform&quantity={adults}&childqty={children}&babyqty={infants}&searchboxarg=t&nonstoponly=off&locale=en-XX&curr=USD"
    raw_trip_results = get_page_content(request_url, wait_for=15000)
    with open("flight_info.html", "w") as f:
        f.write(raw_trip_results)
    def get_flight_list(raw_trip_results: str):
        raw_trip_results = raw_trip_results[raw_trip_results.find("Create Price Alert") + 18:]
        if raw_trip_results.find(r"Get up to 25% off stays by booking a flight, plus free cancellation for your stay if your flight is rescheduled") != -1 and raw_trip_results.find("\"View Details\"") != -1:
            raw_flight_list = raw_trip_results.split(r"Get up to 25% off stays by booking a flight, plus free cancellation for your stay if your flight is rescheduled")
            raw_flight_list = raw_flight_list[:-1]
            for i in range (0, len(raw_flight_list)):
                flight = {
                    "misc": "",
                    "baggage": "",
                    "airlines": "",
                    "price": "",
                    "from": "",
                    "to": "",
                    "departure_time": "",
                    "arrival_time": "",
                    "total_duration": "",
                    "intermediate_stops": ""
                }
                raw_flight = raw_flight_list[i]
                
                if raw_flight.find("Cheapest") != -1:
                    flight['misc'] += "Cheapest "
                    raw_flight = raw_flight.replace("Cheapest", "")
                if raw_flight.find("Fastest") != -1:
                    flight['misc'] += "Fastest "
                    raw_flight = raw_flight.replace("Fastest", "")
                
                if raw_flight.find("Included") != -1:
                    flight['baggage'] = "Carry-on and checked baggage included"
                    raw_flight = raw_flight.replace("Included", "")
                elif raw_flight.find("Carry-on baggage included") != -1:
                    flight["baggage"] = "Carry-on baggage included"
                    raw_flight = raw_flight.replace("Carry-on baggage included", "")
                elif raw_flight.find("Checked baggage included") != -1:
                    flight["baggage"] = "Checked baggage included"
                    raw_flight = raw_flight.replace("Checked baggage included", "")
                else:
                    flight["baggage"] = "Not included"
                    
                if raw_flight.find("CO2e") != -1:
                    raw_flight = raw_flight[raw_flight.find("CO2e") + 4:]
                    
                departure_time_start_index = raw_flight.find(":") - 2
                flight["airlines"] = raw_flight[:departure_time_start_index - 1]
                flight["departure_time"] = raw_flight[departure_time_start_index:departure_time_start_index + 5]
                raw_flight = raw_flight.replace(raw_flight[:departure_time_start_index + 5], "", 1)
                
                if raw_flight[3] == " ":
                    if raw_flight[4] == "T":
                        flight["from"] = raw_flight[:6]
                        raw_flight = raw_flight.replace(flight["from"], "", 1)
                    else:
                        flight["from"] = raw_flight[:5]
                        raw_flight = raw_flight.replace(flight["from"], "", 1)
                else:
                    flight["from"] = raw_flight[:3]
                    raw_flight = raw_flight.replace(flight["from"], "", 1)
                
                if raw_flight[raw_flight.find("h") + 1] == " ":
                    flight["total_duration"] = raw_flight[ :raw_flight.find("m") + 1]
                    raw_flight = raw_flight.replace(flight["total_duration"], "", 1)
                else:
                    flight["total_duration"] = raw_flight[ :raw_flight.find("h") + 1]
                    raw_flight = raw_flight.replace(flight["total_duration"], "", 1)
                    
                arrival_time_start_index = raw_flight.find(":") - 2
                flight["arrival_time"] = raw_flight[arrival_time_start_index : arrival_time_start_index + 5]
                flight["intermediate_stops"] = raw_flight[:arrival_time_start_index]
                raw_flight = raw_flight.replace(flight["intermediate_stops"], "", 1)
                raw_flight = raw_flight.replace(flight["arrival_time"], "", 1)
                
                if raw_flight[3] == " ":
                    if raw_flight[4] == "T":
                        flight["to"] = raw_flight[:6]
                        raw_flight = raw_flight.replace(flight["to"], "", 1)
                    else:
                        flight["to"] = raw_flight[:5]
                        raw_flight = raw_flight.replace(flight["to"], "", 1)
                else:
                    flight["to"] = raw_flight[:3]
                    raw_flight = raw_flight.replace(flight["to"], "", 1)
                
                raw_flight = raw_flight.strip()
                
                if raw_flight[0] == "+":
                    flight["arrival_time"] = flight["arrival_time"] + " " + raw_flight[0:2]
                    raw_flight = raw_flight[2:]
                
                raw_flight = raw_flight.replace("Select", "")
                if raw_flight.find("<") != -1:
                    raw_flight = raw_flight.replace(raw_flight[raw_flight.find("<"):raw_flight.find(" left") + 5], "")
                
                if raw_flight.find("US", 1) != -1:
                    flight["price"] = raw_flight[:raw_flight.find("US", 1)]
                else:
                    flight["price"] = raw_flight
                flight_list.append(flight)

        elif raw_trip_results.find("View Details") != -1:
            raw_flight_list = raw_trip_results.split("View Details")
            raw_flight_list = raw_flight_list[:-1]
            for i in range(0,len(raw_flight_list)):
                flight = {
                    "misc": "",
                    "baggage": "",
                    "airlines": "",
                    "price": "",
                    "from": "",
                    "to": "",
                    "departure_time": "",
                    "arrival_time": "",
                    "total_duration": "",
                    "intermediate_stops": ""
                }
                raw_flight = raw_flight_list[i]
                raw_flight = raw_flight.replace("://", "")
                raw_flight = raw_flight.replace("Recommended: US$", "")
                raw_flight = raw_flight.replace("tickets:", "")
                if raw_flight.find("CO2e") != -1:
                    raw_flight = raw_flight.replace(raw_flight[raw_flight.find(r"\-"):raw_flight.find("CO2e") + 4], "")
                if raw_flight.find("Cheapest") != -1:
                    flight['misc'] += "Cheapest "
                if raw_flight.find("Fastest") != -1:
                    flight['misc'] += "Fastest"
                
                if raw_flight.find("Carry-on and checked baggage included") != -1:
                    flight["baggage"] = "Carry-on and checked baggage included"
                elif raw_flight.find("Carry-on baggage included") != -1:
                    flight["baggage"] = "Carry-on baggage included"
                elif raw_flight.find("Checked baggage included") != -1:
                    flight["baggage"] = "Checked baggage included"
                else:
                    flight["baggage"] = "Not included"
                
                departure_time_start_index = raw_flight.find(":") - 2
                airlines_start_index = raw_flight.rfind("\n", 0, raw_flight.find("...") - 3) + 1
                if raw_flight[airlines_start_index:airlines_start_index + 9] == "Codeshare":
                    airlines_start_index = raw_flight.rfind("\n", 0, airlines_start_index - 3) + 1
                
                flight["airlines"] = raw_flight[airlines_start_index:raw_flight.find("...") - 1]
                
                raw_flight = raw_flight.replace(flight["airlines"], "", 1)
                raw_flight = raw_flight.replace("...", "", 1)
                raw_flight = raw_flight.replace(flight["baggage"], "", 1)
                
                flight["departure_time"] = raw_flight[departure_time_start_index:departure_time_start_index + 5]
                
                
                raw_flight = raw_flight[departure_time_start_index+5:]
                
                raw_flight = raw_flight.replace("\n", "")
                if raw_flight[3] == " ":
                    if raw_flight[4] == "T":
                        flight["from"] = raw_flight[:6]
                        raw_flight = raw_flight.replace(flight["from"], "", 1)
                    else:
                        flight["from"] = raw_flight[:5]
                        raw_flight = raw_flight.replace(flight["from"], "", 1)
                else:
                    flight["from"] = raw_flight[:3]
                    raw_flight = raw_flight.replace(flight["from"], "", 1)
                
                if raw_flight[raw_flight.find("h") + 1] == " ":
                    flight["total_duration"] = raw_flight[ :raw_flight.find("m") + 1]
                    raw_flight = raw_flight.replace(flight["total_duration"], "", 1)
                else:
                    flight["total_duration"] = raw_flight[ :raw_flight.find("h") + 1]
                    raw_flight = raw_flight.replace(flight["total_duration"], "", 1)
                    
                arrival_time_start_index = raw_flight.find(":") - 2
                flight["arrival_time"] = raw_flight[arrival_time_start_index : arrival_time_start_index + 5]
                flight["intermediate_stops"] = raw_flight[:arrival_time_start_index]
                raw_flight = raw_flight.replace(flight["intermediate_stops"], "", 1)
                raw_flight = raw_flight.replace(flight["arrival_time"], "", 1)
                
                if raw_flight[3] == " ":
                    if raw_flight[4] == "T":
                        flight["to"] = raw_flight[:6]
                        raw_flight = raw_flight.replace(flight["to"], "", 1)
                    else:
                        flight["to"] = raw_flight[:5]
                        raw_flight = raw_flight.replace(flight["to"], "", 1)
                else:
                    flight["to"] = raw_flight[:3]
                    raw_flight = raw_flight.replace(flight["to"], "", 1)
                
                raw_flight = raw_flight.strip()
                
                if raw_flight[0] == "+":
                    flight["arrival_time"] = flight["arrival_time"] + " " + raw_flight[0:2]
                    raw_flight = raw_flight[2:]
                
                raw_flight = raw_flight.replace("View Details", "")
                if raw_flight.find("<") != -1:
                    raw_flight = raw_flight.replace(raw_flight[raw_flight.find("<"):raw_flight.find(" left") + 5], "")
                
                if raw_flight.find("US", 1) != -1:
                    flight["price"] = raw_flight[:raw_flight.find("US", 1)]
                else:
                    flight["price"] = raw_flight
                flight_list.append(flight)
        
        elif raw_trip_results.find("Select") != -1 and raw_trip_results.find("...\n") == -1:
            raw_flight_list = raw_trip_results.split("Select")
            raw_flight_list = raw_flight_list[:-1]
            for i in range(0,len(raw_flight_list)):
                flight = {
                    "misc": "",
                    "baggage": "",
                    "airlines": "",
                    "price": "",
                    "from": "",
                    "to": "",
                    "departure_time": "",
                    "arrival_time": "",
                    "total_duration": "",
                    "intermediate_stops": ""
                }
                raw_flight = raw_flight_list[i]
                raw_flight = raw_flight.replace("://", "")
                raw_flight = raw_flight.replace("Recommended: US$", "")
                raw_flight = raw_flight.replace("tickets:", "")
                if raw_flight.find("Cheapest") != -1:
                    flight['misc'] += "Cheapest "
                if raw_flight.find("Fastest") != -1:
                    flight['misc'] += "Fastest"
                
                if raw_flight.find("Included") != -1:
                    flight["baggage"] = "Carry-on and checked baggage included"
                elif raw_flight.find("Carry-on baggage included") != -1:
                    flight["baggage"] = "Carry-on baggage included"
                elif raw_flight.find("Checked baggage included") != -1:
                    flight["baggage"] = "Checked baggage included"
                else:
                    flight["baggage"] = "Not included"
                
                departure_time_start_index = raw_flight.find(":") - 2
                airlines_start_index = raw_flight.rfind("\n", 0, departure_time_start_index - 3) + 1
                if raw_flight[airlines_start_index:airlines_start_index + 9] == "Codeshare":
                    airlines_start_index = raw_flight.rfind("\n", 0, airlines_start_index - 3) + 1
                
                flight["airlines"] = raw_flight[airlines_start_index:departure_time_start_index - 1]
                
                flight["departure_time"] = raw_flight[departure_time_start_index:departure_time_start_index + 5]
                
                raw_flight = raw_flight[departure_time_start_index+5:]
                
                raw_flight = raw_flight.replace("\n", "")
                if raw_flight[3] == " ":
                    if raw_flight[4] == "T":
                        flight["from"] = raw_flight[:6]
                        raw_flight = raw_flight.replace(flight["from"], "", 1)
                    else:
                        flight["from"] = raw_flight[:5]
                        raw_flight = raw_flight.replace(flight["from"], "", 1)
                else:
                    flight["from"] = raw_flight[:3]
                    raw_flight = raw_flight.replace(flight["from"], "", 1)
                
                if raw_flight[raw_flight.find("h") + 1] == " ":
                    flight["total_duration"] = raw_flight[ :raw_flight.find("m") + 1]
                    raw_flight = raw_flight.replace(flight["total_duration"], "", 1)
                else:
                    flight["total_duration"] = raw_flight[ :raw_flight.find("h") + 1]
                    raw_flight = raw_flight.replace(flight["total_duration"], "", 1)
                    
                arrival_time_start_index = raw_flight.find(":") - 2
                flight["arrival_time"] = raw_flight[arrival_time_start_index : arrival_time_start_index + 5]
                flight["intermediate_stops"] = raw_flight[:arrival_time_start_index]
                raw_flight = raw_flight.replace(flight["intermediate_stops"], "", 1)
                raw_flight = raw_flight.replace(flight["arrival_time"], "", 1)
                
                if raw_flight[3] == " ":
                    if raw_flight[4] == "T":
                        flight["to"] = raw_flight[:6]
                        raw_flight = raw_flight.replace(flight["to"], "", 1)
                    else:
                        flight["to"] = raw_flight[:5]
                        raw_flight = raw_flight.replace(flight["to"], "", 1)
                else:
                    flight["to"] = raw_flight[:3]
                    raw_flight = raw_flight.replace(flight["to"], "", 1)
                
                raw_flight = raw_flight.strip()
                
                if raw_flight[0] == "+":
                    flight["arrival_time"] = flight["arrival_time"] + " " + raw_flight[0:2]
                    raw_flight = raw_flight[2:]
                
                raw_flight = raw_flight.replace("Select", "")
                if raw_flight.find("<") != -1:
                    raw_flight = raw_flight.replace(raw_flight[raw_flight.find("<"):raw_flight.find(" left") + 5], "")
                
                if raw_flight.find("US", 1) != -1:
                    flight["price"] = raw_flight[:raw_flight.find("US", 1)]
                else:
                    flight["price"] = raw_flight
                flight_list.append(flight)
        elif raw_trip_results.find("Select") != -1 and raw_trip_results.find("...\n") != -1:
            raw_flight_list = raw_trip_results.split("Select")
            raw_flight_list = raw_flight_list[:-1]
            for i in range(0,len(raw_flight_list)):
                flight = {
                    "misc": "",
                    "baggage": "",
                    "airlines": "",
                    "price": "",
                    "from": "",
                    "to": "",
                    "departure_time": "",
                    "arrival_time": "",
                    "total_duration": "",
                    "intermediate_stops": ""
                }
                raw_flight = raw_flight_list[i]
                raw_flight = raw_flight.replace("://", "")
                raw_flight = raw_flight.replace("Recommended: US$", "")
                raw_flight = raw_flight.replace("tickets:", "")
                if raw_flight.find("CO2e") != -1:
                    raw_flight = raw_flight.replace(raw_flight[raw_flight.find(r"\-"):raw_flight.find("CO2e") + 4], "")
                if raw_flight.find("Cheapest") != -1:
                    flight['misc'] += "Cheapest "
                if raw_flight.find("Fastest") != -1:
                    flight['misc'] += "Fastest"
                
                if raw_flight.find("Carry-on and checked baggage included") != -1:
                    flight["baggage"] = "Carry-on and checked baggage included"
                elif raw_flight.find("Carry-on baggage included") != -1:
                    flight["baggage"] = "Carry-on baggage included"
                elif raw_flight.find("Checked baggage included") != -1:
                    flight["baggage"] = "Checked baggage included"
                else:
                    flight["baggage"] = "Not included"
                
                departure_time_start_index = raw_flight.find(":") - 2
                airlines_start_index = raw_flight.rfind("\n", 0, raw_flight.find("...") - 3) + 1
                if raw_flight[airlines_start_index:airlines_start_index + 9] == "Codeshare":
                    airlines_start_index = raw_flight.rfind("\n", 0, airlines_start_index - 3) + 1
                
                flight["airlines"] = raw_flight[airlines_start_index:raw_flight.find("...") - 1]
                
                raw_flight = raw_flight.replace(flight["airlines"], "", 1)
                raw_flight = raw_flight.replace("...", "", 1)
                raw_flight = raw_flight.replace(flight["baggage"], "", 1)
                
                flight["departure_time"] = raw_flight[departure_time_start_index:departure_time_start_index + 5]
                
                
                raw_flight = raw_flight[departure_time_start_index+5:]
                
                raw_flight = raw_flight.replace("\n", "")
                if raw_flight[3] == " ":
                    if raw_flight[4] == "T":
                        flight["from"] = raw_flight[:6]
                        raw_flight = raw_flight.replace(flight["from"], "", 1)
                    else:
                        flight["from"] = raw_flight[:5]
                        raw_flight = raw_flight.replace(flight["from"], "", 1)
                else:
                    flight["from"] = raw_flight[:3]
                    raw_flight = raw_flight.replace(flight["from"], "", 1)
                
                if raw_flight[raw_flight.find("h") + 1] == " ":
                    flight["total_duration"] = raw_flight[ :raw_flight.find("m") + 1]
                    raw_flight = raw_flight.replace(flight["total_duration"], "", 1)
                else:
                    flight["total_duration"] = raw_flight[ :raw_flight.find("h") + 1]
                    raw_flight = raw_flight.replace(flight["total_duration"], "", 1)
                    
                arrival_time_start_index = raw_flight.find(":") - 2
                flight["arrival_time"] = raw_flight[arrival_time_start_index : arrival_time_start_index + 5]
                flight["intermediate_stops"] = raw_flight[:arrival_time_start_index]
                raw_flight = raw_flight.replace(flight["intermediate_stops"], "", 1)
                raw_flight = raw_flight.replace(flight["arrival_time"], "", 1)
                
                if raw_flight[3] == " ":
                    if raw_flight[4] == "T":
                        flight["to"] = raw_flight[:6]
                        raw_flight = raw_flight.replace(flight["to"], "", 1)
                    else:
                        flight["to"] = raw_flight[:5]
                        raw_flight = raw_flight.replace(flight["to"], "", 1)
                else:
                    flight["to"] = raw_flight[:3]
                    raw_flight = raw_flight.replace(flight["to"], "", 1)
                
                raw_flight = raw_flight.strip()
                
                if raw_flight[0] == "+":
                    flight["arrival_time"] = flight["arrival_time"] + " " + raw_flight[0:2]
                    raw_flight = raw_flight[2:]
                
                raw_flight = raw_flight.replace("Select", "")
                if raw_flight.find("<") != -1:
                    raw_flight = raw_flight.replace(raw_flight[raw_flight.find("<"):raw_flight.find(" left") + 5], "")
                
                if raw_flight.find("US", 1) != -1:
                    flight["price"] = raw_flight[:raw_flight.find("US", 1)]
                else:
                    flight["price"] = raw_flight
                flight_list.append(flight)
                
        else:
            return []
    flight_list = get_flight_list(raw_trip_results)
    # print(raw_trip_results)
    
    return QueryContextPage(url=request_url, title=f"Flight information from {from_city} ({from_airport}) to {to_city} ({to_airport}) on {date}", content="\n".join(str(item) for item in flight_list))

"""
https://www.trip.com/flights/showfarefirst?dcity=ctu&acity=muc&ddate=2025-07-12&rdate=2025-07-15&dairport=tfu&aairport=muc&triptype=ow&class=y&lowpricesource=searchform&quantity=1&childqty=1&babyqty=1&searchboxarg=t&nonstoponly=off&locale=en-XX&curr=USD
"""

def get_lonelyplanet_articles(experiment_id, limit: int = 10):
    def clean_content(content: str):
        content = content.replace("\nBack\n", "\n")
        content = content.replace("\nDestinations\n", "\n")
        content = content.replace("\nBooks\n", "\n")
        content = content.replace("\nTrips\n", "\n")
        content = content.replace("\nStories\n", "\n")
        content = content.replace("\nAccount\n", "\n")
        content = content.replace("\nAdvertisement\n", "\n")
        content = content.replace("[Cart](https://shop.lonelyplanet.com/cart?utm_source=lonelyplanet&utm_campaign=lpcart)\n", "")
        while content.find("lp-cms-production.imgix.net") != -1:
            index = content.find("lp-cms-production.imgix.net")
            start_index = content.rfind("\n", 0, index)
            end_index = content.find("\n", index)
            content = content.replace(content[start_index:end_index], "")
        while content.find("](http") != -1:
            index = content.find("](http")
            start_index = content.rfind("[", 0, index)
            end_index = content.find(")", index)
            content = content.replace(content[start_index:end_index + 1], content[start_index + 1:index])
        if content.find("## Get a book. Get inspired. Get exploring.") != -1:
            content = content[:content.find("## Get a book. Get inspired. Get exploring.")].strip()
        else:
            index = content.find("\n$")
            while content[content.find("\n", index) - 3:content.find("\n", index)] != "USD":
                if index == -1:
                    index = len(content)
                    break
                index = content.find("\n", index + 1)
            content = content[:index].strip()
        return content
    article_list = []
    url_list = []
    for i in range(1, 1000):
        if len(url_list) >= limit:
            break
        raw_results = get_page_content(f"https://www.lonelyplanet.com/articles?page={i}")
        index = raw_results.find("https://www.lonelyplanet.com/articles/")
        while index != -1:
            end_index = raw_results.find(')', index)
            url_list.append(raw_results[index:end_index])
            index = raw_results.find("https://www.lonelyplanet.com/articles/", end_index)
    for url in url_list:
        if len(article_list) >= limit:
            break
        article_list.append(QueryContextPage(title=url.replace("https://www.lonelyplanet.com/articles/", ""), url = url, content = clean_content(get_page_content(url))))
    with open(f"experiments/{str(experiment_id)}/web_contents/{time.strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        f.write(json.dumps([page.json() for page in article_list], indent=4))


def get_wikipedia_random_pages(experiment_id, limit: int = 10):
    # raw_results = requests.get("https://en.wikipedia.org/w/api.php?format=json&action=query&generator=random&grnnamespace=0&prop=revisions|images&rvprop=content&grnlimit=10")
    def clean_content(content: str):
        content = content.replace("<br>", " ")
        while content.find("](http") != -1:
            index = content.find("](http")
            start_index = content.rfind("[", 0, index)
            end_index = content.find(")", index)
            content = content.replace(content[start_index:end_index + 1], content[start_index + 1:index])
        
        index = content.find("[\\")
        while index != -1:
            end_index = content.find("\\]", index)
            if end_index == -1:
                break
            to_be_deleted = True
            for char in content[index + 2:end_index]:
                if not char.isdigit():
                    to_be_deleted = False
                    break
            if to_be_deleted:
                content = content[:index] + content[end_index + 2:]
                index = content.find("[\\", index)
            else:
                index = content.find("[\\", index + 1)

        return content
    url_list = []
    page_list = []
    while len(url_list) < limit:
        try:
            for i in range(10):
                raw_results = requests.get("https://en.wikipedia.org/api/rest_v1/page/random/summary")
                if raw_results.status_code == 200:
                    break
            data = raw_results.json()
            url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
            if url and url not in url_list:
                print(f"Found URL: {url}")
                raw_content = get_page_content(url, wait_for=5000)
                content = clean_content(raw_content)
                if len(content) >= 6400 and len(content) <= 100000:
                    url_list.append(url)
                    title = data.get("title", "Untitled")
                    page_list.append(QueryContextPage(title=title, url=url, content=content))
                    print(f"Added page: {title}")
                    
            time.sleep(1)
        except Exception as e:
            print(e)
            time.sleep(1)
    with open(f"experiments/{str(experiment_id)}/web_contents/{time.strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        f.write(json.dumps([page.json() for page in page_list], indent=4))

def get_petapixel_articles(experiment_id, limit: int = 10):
    def clean_content(content: str):
        index = content.find("](http")
        while index != -1:
            start_index = content.rfind("[", 0, index)
            if start_index > 0 and content[start_index - 1] == "!":
                start_index = start_index - 1
            end_index = content.find(")", index)
            content = content.replace(content[start_index:end_index + 1], content[start_index + 1:index])
            index = content.find("](http", index)
        #delete all lines that does not end with a punctuation mark
        lines = content.split("\n")
        cleaned_lines = []
        for line in lines:
            if line.strip() and (not (line.strip()[-1].isalpha() or line.strip()[-1].isdigit())):
                cleaned_lines.append(line.strip())
        content = "\n".join(cleaned_lines)
        content = content.replace("PetaPixel articles may include affiliate links; if you buy something through such a link, PetaPixel may earn a commission.", "")
        return content
    article_list = []
    url_list = []
    for i in range(1, 1000):
        if len(url_list) >= limit:
            break
        raw_results = get_page_content(f"https://petapixel.com/topic/reviews/page/{i}/")
        index = raw_results.find("https://petapixel.com/20")
        while index != -1:
            end_index = raw_results.find(')', index)
            url = raw_results[index:end_index]
            if url not in url_list:
                url_list.append(url)
            index = raw_results.find("https://petapixel.com/20", end_index)
    for url in url_list:
        if len(article_list) >= limit:
            break
        article_list.append(QueryContextPage(title=url[len("https://petapixel.com/2025/07/28/"):-1], url = url, content = clean_content(get_page_content(url))))
        print(article_list[-1].title)
        print(article_list[-1].url)
        # print(article_list[-1].content)
    
    with open(f"experiments/{str(experiment_id)}/web_contents/{time.strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        f.write(json.dumps([page.json() for page in article_list], indent=4))


def get_pitchfork_articles(experiment_id, limit: int = 10):
    def clean_content(content: str):
        index = content.find("](http")
        while index != -1:
            start_index = content.rfind("[", 0, index)
            if start_index > 0 and content[start_index - 1] == "!":
                start_index = start_index - 1
            end_index = content.find(")", index)
            content = content.replace(content[start_index:end_index + 1], content[start_index + 1:index])
            index = content.find("](http", index)
        #delete all lines that does not end with a punctuation mark
        lines = content.split("\n")
        cleaned_lines = []
        for line in lines:
            if line.strip() and (not (line.strip()[-1].isalpha() or line.strip()[-1].isdigit() or line.strip()[-1] == "_")):
                cleaned_lines.append(line.strip())
        content = "\n".join(cleaned_lines)
        content = content.replace("All products featured on Pitchfork are independently selected by our editors. However, when you buy something through our retail links, we may earn an affiliate commission.", "")
        return content
    
    article_list = []
    url_list = []
    for i in range(1, 1000):
        if len(url_list) >= limit:
            break
        raw_results = get_page_content(f"https://pitchfork.com/reviews/albums/?page={i}")
        index = raw_results.find("https://pitchfork.com/reviews/albums/")
        while index != -1:
            end_index = raw_results.find(')', index)
            if raw_results[index + len("https://pitchfork.com/reviews/albums/")] == "?":
                index = raw_results.find("https://pitchfork.com/reviews/albums/", end_index)
                continue
            url = raw_results[index:end_index]
            if url not in url_list:
                url_list.append(url)
            index = raw_results.find("https://pitchfork.com/reviews/albums/", end_index)
    for url in url_list:
        if len(article_list) >= limit:
            break
        article_list.append(QueryContextPage(title=url[len("https://pitchfork.com/reviews/albums/"):-1], url = url, content = clean_content(get_page_content(url))))
        print(article_list[-1].title)
        print(article_list[-1].url)
        print(article_list[-1].content)
    
    with open(f"experiments/{str(experiment_id)}/web_contents/{time.strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        f.write(json.dumps([page.json() for page in article_list], indent=4))

def get_olh_journals(experiment_id, limit = 3):
    def extract_abstract_introduction(content: str):
        # print(content)
        abstract_start_index = content.find("## Abstract")
        abstract_end_index = content.find("\n#", abstract_start_index)
        abstract = ""
        if abstract_start_index != -1 and abstract_end_index != -1:
            abstract = content[abstract_start_index:abstract_end_index].strip()
        introduction_start_index = -1
        index = content.find("## ")
        while index != -1:
            if content.find("Introduction", index) != -1 and (content.find("\n", index) > content.find("Introduction", index)):
                introduction_start_index = index
                break
            index = content.find("## ", index + 1)
        introduction_end_index = content.find("\n#", introduction_start_index)
        introduction = ""
        if introduction_start_index != -1 and introduction_end_index != -1:
            introduction = content[introduction_start_index:introduction_end_index].strip()
        print(introduction)
        print(abstract)
        if introduction == "" or abstract == "":
            return ""
        return abstract + "\n\n" + introduction

    article_list = []
    url_list = []
    for i in range(1, 1000):
        if len(url_list) >= limit:
            break
        raw_results = get_page_content(f"https://olh.openlibhums.org/articles/?order_by=-date_published&page={i}&paginate_by=100")
        index = raw_results.find("https://olh.openlibhums.org/article/id/")
        while index != -1:
            end_index = raw_results.find(')', index)
            url = raw_results[index:end_index]
            if url not in url_list and url.find("/file/") == -1:
                url_list.append(url)
            index = raw_results.find("https://olh.openlibhums.org/article/id/", end_index)
    # print(url_list)
    for url in url_list:
        print(url)
        if len(article_list) >= limit:
            break
        page_content = get_page_content(url)
        title = page_content[page_content.find("![") + 2:page_content.find("](", page_content.find("![") + 2)]
        abstract_introduction = remove_links_from_markdown(extract_abstract_introduction(page_content))
        if abstract_introduction == "":
            continue
        article_list.append(QueryContextPage(title=title, url=url, content=abstract_introduction))
        print(article_list[-1].title)
        print(article_list[-1].url)
        print(article_list[-1].content)
    
    with open(f"experiments/{str(experiment_id)}/web_contents/{time.strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        f.write(json.dumps([page.json() for page in article_list], indent=4))

def get_zhihu_trending():
    chrome_options=Options()
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get("https://www.zhihu.com")
    convert_cookies_to_python()
    from cookies_data import COOKIES_LIST
    for cookie in COOKIES_LIST:
        driver.add_cookie(cookie)
    driver.get("https://www.zhihu.com/hot")
    html = driver.execute_script("return document.documentElement.outerHTML")
    soup = BeautifulSoup(html, "html.parser")
    with open("zhihu_trending.html", "w", encoding="utf-8") as f:
        f.write(str(soup.prettify()))
    hot_item_list = soup.find_all("div", {"class": "HotItem-content"})
    return_list = []
    for item in hot_item_list:
        title = item.find("h2", {"class": "HotItem-title"})
        if title is None:
            continue
        title = title.get_text(strip=True)
        url = item.find("a", {"title": title})
        if url is None:
            continue
        url = url['href']
        content = item.find("p", {"class": "HotItem-excerpt"})
        if content is None:
            continue
        content = content.get_text(strip=True)
        return_list.append(QueryContextPage(title=title, url=url, content=content))
        print(f"Title: {title}\nURL: {url}\nContent: {content}\n")
    return return_list
        
def get_cnn_news(experiment_id, topic: Union[None, Literal["us", "world", "politics", "business", "health", "entertainment", "style", "travel", "sports", "climate", "science", "weather", "games", "all"]] = None, limit=10, url_list: list[str] = None):
    if url_list is not None:
        news_url_list = url_list
    elif topic == "all":
        news_url_list = []
        for t in ["us", "world", "politics", "business", "health", "entertainment", "style", "travel", "sports", "climate", "science", "weather", "games"]:
        # for t in ["2023-7"]:
            raw_cnn_news = get_page_content(f"https://edition.cnn.com/{t}")
            # raw_cnn_news = get_page_content(f"https://edition.cnn.com/article/sitemap-{t}.html")
            index = raw_cnn_news.find("(https://edition.cnn.com/20") + 1
            for i in range(0, limit):
                if index == 0:
                    break
                news_url = raw_cnn_news[index:raw_cnn_news.find(")", index)]
                if not news_url.find("/video/") != -1:
                    news_url_list.append(news_url)
                index = raw_cnn_news.find("(https://edition.cnn.com/20", index + 1) + 1
    else:
        raw_cnn_news = get_page_content(f"https://edition.cnn.com/{topic if topic else ''}")
        news_url_list = []
        index = raw_cnn_news.find("(https://edition.cnn.com/20") + 1
        for i in range(0, limit):
            if index == 0:
                break
            news_url = raw_cnn_news[index:raw_cnn_news.find(")", index)]
            if not news_url.find("/video/") != -1:
                news_url_list.append(news_url)
            index = raw_cnn_news.find("(https://edition.cnn.com/20", index + 1) + 1
    news_url_list = list(set(news_url_list))
    print(len(news_url_list))
    news_list = []
    def clean_content(content: str) -> str:
        content = content[content.find("\n# ") + 1:]
        index = content.find("FacebookTweet")
        while index != -1:
            end_index = content.find("Link Copied!\n", index) + 13
            content = content.replace(content[index:end_index], "")
            index = content.find("FacebookTweet")
        index = content.find("## Up next")
        if index != -1:
            content = content[:index]
        index = content.find("## Most read")
        if index != -1:
            content = content[:index]
        index = content.find("](mailto:")
        while index != -1:
            start_index = content.rfind("[", 0, index)
            end_index = content.find(")", index)
            content = content.replace(content[start_index:end_index + 1], "")
            index = content.find("](mailto:", index + 1)
        index = content.find("![")
        while index != -1:
            caption = content[index + 2:content.find("]", index)]
            end_index = content.find(")", content.find("]", index))
            content = content.replace(content[index:end_index + 1], "")
            content = content.replace(caption,"")
            index = content.find("![")
        index = content.find("](https://edition.cnn.com/")
        while index != -1:
            start_index = content.rfind("[", 0, index)
            end_index = content.find(")", index)
            content = content.replace(content[start_index:end_index + 1], "")
            index = content.find("](https://edition.cnn.com/", index + 1)
        index = content.find("](http")
        while index != -1:
            start_index = content.rfind("[", 0, index)
            end_index = content.find(")", index)
            content = content.replace(content[start_index:end_index + 1], content[start_index + 1:index])
            index = content.find("](http", index + 1)
        content = content.replace("Ad Feedback\n", "")
        content = content.replace("Link Copied!\n", "")
        content = content.replace("Follow:\n", "")
        content = content.replace("\ninfo\n\nThe top numbered articles will be viewable to readers when programmed onto a page. Make sure you have 30 or more cards added to the container for the machine learning (ML) feature to work. Your queued up articles, seen in the violet section, will use ML to surface up relevant content to the reader based on their individual interests and preferences.", "")
        index = content.find("\n")
        while index != -1:
            if content[index - 1].isalpha() or content[index - 1].isdigit():
                to_be_deleted = content[max(0, content.rfind("\n", 0, index)):index + 1]
                if not (to_be_deleted.startswith("\n#") or to_be_deleted.startswith("#")):
                    index = index - len(to_be_deleted)
                    content = content.replace(to_be_deleted, "")
            index = content.find("\n", index + 1)
        while content.find("\n\n\n") != -1:
            content = content.replace("\n\n\n", "\n\n")
        return content
    for url in news_url_list:
        content = get_page_content(url, timeout=300000)
        content = clean_content(content)
        title = content[content.find("# ") + 2:content.find("\n", content.find("# ") + 2)]
        news_list.append(QueryContextPage(title=title, url=url, content=content))
    with open(f"experiments/{str(experiment_id)}/web_contents/{time.strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        f.write(json.dumps([news.json() for news in news_list], indent=4))
    return news_list
        

if __name__ == "__main__":
    # get_arxiv_abstract_and_introduction("303_arxiv_oai", ["cs"], 35)
    get_olh_journals("000_test", 35)