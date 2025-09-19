import requests
from bs4 import BeautifulSoup
import re
import json
from urllib.parse import urlparse, urljoin
import time
import os
from tqdm import tqdm

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

def get_subpage_urls(main_url):
    '''
    Obtains all valid subpage links from all paginations.
    '''
    base_domain = urlparse(main_url).netloc
    visited_pages = set()
    subpage_urls = set()
    page_queue = [main_url]

    while page_queue:
        current_url = page_queue.pop(0)
        if current_url in visited_pages:
            continue
        visited_pages.add(current_url)

        try:
            response = requests.get(current_url, headers=headers, timeout=10)
            if response.status_code != 200:
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Fetch all valid subpage links
            for a in soup.find_all('a', href=True):
                href = a['href'].split('#')[0]
                full_url = urljoin(current_url, href)
                parsed = urlparse(full_url)
                
                if parsed.netloc != base_domain:
                    continue
                if '/pg/' in full_url:
                    if full_url not in visited_pages and full_url not in page_queue:
                        page_queue.append(full_url)
                    continue
                
                path_parts = parsed.path.strip('/').split('/')
                if len(path_parts) >= 3 and path_parts[0] == 'minecraft':
                    subpage_urls.add(full_url)

            # Deal with pagination links
            pagination_links = soup.select('a[href*="/pg/"]')
            for a in pagination_links:
                href = a['href'].split('#')[0]
                full_url = urljoin(current_url, href)
                if full_url not in visited_pages and full_url not in page_queue:
                    page_queue.append(full_url)

        except Exception as e:
            print(f"Error processing {current_url}: {str(e)}")
        
        time.sleep(1)

    return list(subpage_urls)


def extract_building_name(url):
    '''
    Extract the building name from the URLs.
    '''
    parsed = urlparse(url)
    path_parts = parsed.path.strip('/').split('/')
    return path_parts[2] if len(path_parts) >= 3 else "unknown"


def download_image(url, save_dir):
    '''
    Download and save images.
    '''
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            filename = os.path.basename(urlparse(url).path)
            save_path = os.path.join(save_dir, filename)
            
            if not os.path.exists(save_path):
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"Image saved: {save_path}")
                return filename
            return None
    except Exception as e:
        print(f"Failed to download image {url}: {str(e)}")
    return None


def scrape_subpage(sub_url):
    '''
    Scrape data from a single subpage.
    '''
    try:
        response = requests.get(sub_url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, None, []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract building name
        title_tag = soup.find('h1')
        building_name = re.sub(r'[\\/*?:"<>|]', '', title_tag.get_text(strip=True)) if title_tag else extract_building_name(sub_url)
        building_name = building_name.replace(' ', '_').lower()[:50]

        # Extract image URLs
        image_urls = []
        for img in soup.find_all('img', src=True):
            if '/files/products/medium/' in img['src']:
                full_url = urljoin(sub_url, img['src'])
                image_urls.append(full_url)

        # Get the LayerMap JavaScript URL
        layer_js_url = None
        for script in soup.find_all('script', src=True):
            if '/js/LayerMap/LayerMap_' in script['src']:
                layer_js_url = urljoin(sub_url, script['src'])
                break
        
        layer_data = {}
        if layer_js_url:
            try:
                js_response = requests.get(layer_js_url, headers=headers, timeout=10)
                if js_response.status_code == 200:
                    match = re.search(r'\{.*\}', js_response.text, re.DOTALL)
                    if match:
                        layer_data = json.loads(match.group(0))
            except Exception as e:
                print(f"Error loading layer data: {str(e)}")

        # Extract properties table
        properties = {}
        props_table = soup.find('table', id='object_properties')
        if props_table:
            for row in props_table.find_all('tr'):
                cols = row.find_all(['th', 'td'])
                if len(cols) == 2:
                    key = cols[0].get_text(strip=True)
                    value = cols[1].get_text(strip=True)
                    properties[key] = value

        return {
            'metadata': {
                'url': sub_url,
                'title': title_tag.get_text(strip=True) if title_tag else "",
                # 'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            },
            'layer_data': layer_data,
            'properties': properties,
            'image_urls': image_urls
        }, building_name, image_urls

    except Exception as e:
        print(f"Error scraping {sub_url}: {str(e)}")
        return None, None, []


def main():
    all_urls = ['https://www.grabcraft.com/minecraft/castles', 'https://www.grabcraft.com/minecraft/ruins', 'https://www.grabcraft.com/minecraft/houses', 'https://www.grabcraft.com/minecraft/churches', 'https://www.grabcraft.com/minecraft/famous-films', 'https://www.grabcraft.com/minecraft/fictional', 'https://www.grabcraft.com/minecraft/sightseeing-buildings', 'https://www.grabcraft.com/minecraft/farm-buildings', 'https://www.grabcraft.com/minecraft/military-buildings', 'https://www.grabcraft.com/minecraft/parks', 'https://www.grabcraft.com/minecraft/gardens', 'https://www.grabcraft.com/minecraft/bridges', 'https://www.grabcraft.com/minecraft/roads', 'https://www.grabcraft.com/minecraft/other-190', 'https://www.grabcraft.com/minecraft/items', 'https://www.grabcraft.com/minecraft/animals', 'https://www.grabcraft.com/minecraft/fictional-characters', 'https://www.grabcraft.com/minecraft/famous-characters', 'https://www.grabcraft.com/minecraft/fictional-characters-169', 'https://www.grabcraft.com/minecraft/items-170', 'https://www.grabcraft.com/minecraft/miscellaneous-171', 'https://www.grabcraft.com/minecraft/famous-characters-181', 'https://www.grabcraft.com/minecraft/cars', 'https://www.grabcraft.com/minecraft/ships', 'https://www.grabcraft.com/minecraft/emergency-vehicles', 'https://www.grabcraft.com/minecraft/planes', 'https://www.grabcraft.com/minecraft/buses', 'https://www.grabcraft.com/minecraft/working-vehicles', 'https://www.grabcraft.com/minecraft/other-transportation', 'https://www.grabcraft.com/minecraft/spaceships', 'https://www.grabcraft.com/minecraft/machines', 'https://www.grabcraft.com/minecraft/items-188', 'https://www.grabcraft.com/minecraft/secret-passageways']

    for i in tqdm(range(len(all_urls))):
        main_url = all_urls[i].strip()
        
        # Create category directory
        parsed_main = urlparse(main_url)
        category = parsed_main.path.strip('/').split('/')[-1]
        output_dir = os.path.join('./raw_html_data', category)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Fetch all subpage links
        print("Starting to collect pagination links...")
        subpages = get_subpage_urls(main_url)
        print(f"Discovered {len(subpages)} valid subpages.")
        
        # Deal with each subpage
        success_count = 0
        for idx, url in enumerate(subpages, 1):
            print(f"\nDealing with ({idx}/{len(subpages)}) {url}")
            data, building_name, image_urls = scrape_subpage(url)
            
            if data and building_name:
                # Create building-specific directory
                building_dir = os.path.join(output_dir, building_name)
                if not os.path.exists(building_dir):
                    os.makedirs(building_dir, exist_ok=True)
                else:
                    print('already existed!')
                    continue
                
                timestamp = str(int(time.time()))
                json_filename = f"{building_name}_{timestamp}.json"
                json_path = os.path.join(building_dir, json_filename)
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"JSON saved: {json_path}")
                
                if image_urls:
                    print(f"There are {len(image_urls)} images to be downloaded.")
                    for img_url in image_urls:
                        download_image(img_url, building_dir)
                
                success_count += 1
            
            time.sleep(2)

        print(f"\nSuccess! Has addressed {success_count}/{len(subpages)} items.")

if __name__ == '__main__':
    main()