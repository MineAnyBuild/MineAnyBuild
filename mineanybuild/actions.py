from javascript import require, On, Once, AsyncTask, once, off
import math
import time
from math import cos, sin, radians
# Import the javascript libraries
mineflayer = require("mineflayer")
pathfinder = require('mineflayer-pathfinder')
vec3 = require("vec3")
import asyncio

class Actions:
    def __init__(self, bot, mcData, movements,playerUsername):
        self.bot = bot
        self.mcData = mcData
        self.movements = movements
        self.playerUsername = playerUsername

    def log(self, message):
        print(f"[{self.bot.username}] {message}")
    
    def setControlState(self, control, state):
        if self.bot:
            self.bot.setControlState(control, state)
        else:
            print("Bot is not created yet. Call createBot() first.")

    def move(self, direction, blocks):
        if direction not in ["forward", "back", "left", "right"]:
            print(f"Invalid move direction: {direction}")
            return

        startPosition = self.bot.entity.position.clone()
        endPosition = startPosition.clone()

        # Calculate the end position based on the direction and blocks
        if direction == "forward":
            endPosition.z -= blocks
        elif direction == "back":
            endPosition.z += blocks
        elif direction == "left":
            endPosition.x -= blocks
        elif direction == "right":
            endPosition.x += blocks

        # Call pathfinding to move to the end position
        self.pathfinding(endPosition)

    def forward(self, blocks):
        self.move("forward", blocks)

    def back(self, blocks):
        self.move("back", blocks)

    def left(self, blocks):
        self.move("left", blocks)

    def right(self, blocks):
        self.move("right", blocks)

    def jump(self, state=True):
        self.setControlState("jump", state)

    def stop(self):
        # Stops all movement
        self.setControlState("forward", False)
        self.setControlState("back", False)
        self.setControlState("left", False)
        self.setControlState("right", False)
        self.setControlState("jump", False)

    

    def pathfinding(self, goalPosition, findPlayer=None):
        if findPlayer:
            localPlayers = self.bot.players
            target = None
            for player in localPlayers:
                playerData = localPlayers[player]
                if playerData["username"] == self.playerUsername:
                    target = localPlayers[player].entity
                    
            if not target:
                self.bot.chat("I don't see you!")
                return
            
            goalPosition = target.position
            self.bot.pathfinder.setGoal(pathfinder.goals.GoalNear(goalPosition.x, goalPosition.y, goalPosition.z, 1))
        else:
            attempts = 0
            while attempts < 3:
                try:
                    self.bot.pathfinder.setGoal(pathfinder.goals.GoalNear(goalPosition.x, goalPosition.y, goalPosition.z, 1))
                    break  # If successful, exit the loop
                except Exception as e:
                    print(str(e))
                    attempts += 1
                    self.bot.chat(" Refinding path...")
                    if attempts >= 3:
                        self.bot.chat("I couldn't set the goal. Please try again.")

    

    def equip(self, name, destination):
        item = self.findInventoryItem(name, max_attempts=3)
            
        if not item:
            raise ValueError(f"No item named {item} in inventory")
        self.bot.equip(item, destination)


    def consume(self):
         self.bot.consume()
    
    def fish(self):
         self.bot.fish()
    
    def sleep(self,bedBlock):
         self.bot.sleep(bedBlock)
    
    def activateBlock(self,block): #  This is the same as right-clicking a block in the game. Useful for buttons, doors, etc. You must get to the block first
         self.bot.activateBlock(block)
    
    def lookAt(self,position): #  Look at the specified position. You must go near the position before you look at it. To fill bucket with water, you must lookAt first. `position` is `Vec3`
         self.bot.lookAt(position)
    
    def useOn(self,entity): # This is the same as right-clicking an entity in the game. Useful for shearing sheep, equipping harnesses, etc.
         self.bot.useOn(entity)

    def activateItem(self):
        self.bot.activateItem()
    

    def placeItem(self, name, direction):
        if not isinstance(name, str):
            raise ValueError("name for placeItem must be a string")

        itemByName = self.mcData.itemsByName[name]
        if not itemByName:
            raise ValueError(f"No item named {name}")

        item = self.bot.inventory.findInventoryItem(itemByName['id'])
        if not item:
            self.bot.chat(f"No {name} in inventory")
            return
        else:
            self.bot.chat(f"Found {name} in inventory")


        directionVectors = {
            "up": vec3(0, 1, 0),
            "down": vec3(0, -1, 0),
            "front": vec3(1, 0, 0),
            "back": vec3(-1, 0, 0),
            "left": vec3(0, 0, 1),
            "right": vec3(0, 0, -1),
        }

        if direction not in directionVectors:
            raise ValueError(f"Invalid direction {direction}. Choose from 'up', 'down', 'front', 'back', 'left', 'right'.")


        directions_to_try = [direction] + [dir for dir in directionVectors if dir != direction]

        for dir in directions_to_try:
            try:
                faceVector = directionVectors[dir]
                agentPosition = self.bot.entity.position
                targetPosition = agentPosition.offset(faceVector.x, faceVector.y, faceVector.z)
                print(f"Trying to place {name} in direction {dir}")

                if dir in ['up', 'down']:
                    referenceBlock = self.bot.blockAt(targetPosition)
                else:
                    blockBelowTarget = targetPosition.offset(0, -1, 0)
                    referenceBlock = self.bot.blockAt(blockBelowTarget)

                if referenceBlock and referenceBlock.name != "air":
                    self.bot.chat(f"Placing {name} on {referenceBlock.name}")
                else:
                    self.bot.chat(f"No valid block to place {name} on in direction {dir}. Trying next direction.")
                    continue

                self.equip(item, "hand")
                self.bot.placeBlock(referenceBlock, vec3(0, 1, 0))
                self.bot.chat(f"The block {name} has been successfully placed in direction {dir}.")
                break  # Exit the loop if the item is placed successfully

            except Exception as e:
                self.bot.chat(f"Failed to place {name} in direction {dir}. Trying next direction.")
                continue
        else:
            self.bot.chat(f"Failed to place {name} in all attempted directions. Please check the surroundings and try again.")


    def setInventorySlot(self, itemName, count):
        # print(itemName)
        try:
            itemID = self.bot.registry.itemsByName[itemName].id
        except Exception as e:
            raise ValueError(f"Could not get {itemName}")
        Item = require('prismarine-item')('1.18')
        attempts = 0
        success = False

        while attempts < 3 and not success:
            try:
                self.bot.creative.setInventorySlot(36, Item(itemID, count))
                self.bot.chat(f"Acquired {itemName} at slot 36")
                success = True  # Set flag to true if operation succeeds
            except Exception as e:
                attempts += 1
                self.bot.chat(f" Reattempt acquiring {itemName}")
                if attempts == 3:
                    raise ValueError(f"Could not get {itemName}")
        return success



    def findInventoryItem(self, itemName, max_attempts=3):
        """
        Attempts to find and set the inventory slot with the specified block type.
        Retries up to max_attempts times if it fails.
        
        Parameters:
        - blockType: The type of block to set in the inventory.
        - max_attempts: The maximum number of attempts to find the item in the inventory.
        
        Returns:
        - True if successful, False if it fails after max_attempts.
        """
        for attempt in range(max_attempts):
            try:
                itemByName = self.bot.registry.itemsByName[itemName]
                item = self.bot.inventory.findInventoryItem(itemByName['id'])
                if item:
                    return item
                else:
                    self.bot.chat(f"Attempt {attempt + 1}: {itemName} not found in inventory.")
            except Exception as e:
                # self.bot.chat(f"reattempt finding {itemName}")
                print(f"reattempt finding {itemName}")



    def placeBlockAt(self, name, position):

        item = self.findInventoryItem(name, max_attempts=3)
        if not item:
            self.bot.chat(f"No {name} in inventory")
            return
        
        # Check if there's already a block at the target position and destroy it if necessary
        targetBlock = self.bot.blockAt(position)
        # print(targetBlock.name)
        if targetBlock and targetBlock.name != "air" and name != "water_bucket":
            self.bot.chat(f"Block at X: {position.x:.2f}, Y: {position.y:.2f}, Z: {position.z:.2f} is not air, destroying it first.")
            self.bot.dig(targetBlock)
            # yield self.bot.waitForDiggingCompletion()  # Ensure the block is completely destroyed

        # Define the surrounding positions (reference blocks) and corresponding face vectors
        referenceBlocks = {
            "down": (position.offset(0, -1, 0), vec3(0, 1, 0)),
            "up": (position.offset(0, 1, 0), vec3(0, -1, 0)),
            "north": (position.offset(0, 0, -1), vec3(0, 0, 1)),
            "south": (position.offset(0, 0, 1), vec3(0, 0, -1)),
            "west": (position.offset(-1, 0, 0), vec3(1, 0, 0)),
            "east": (position.offset(1, 0, 0), vec3(-1, 0, 0))
        }
        # Try to place the block from each reference block
        # Try to place the block from each reference block
        for direction, (refBlockPos, faceVector) in referenceBlocks.items():
            referenceBlock = self.bot.blockAt(refBlockPos)
            if referenceBlock and referenceBlock.name != "air":
                if abs(self.bot.entity.position.y - position.y) > 5:
                    self.jump_and_place_until(position, 'dirt') 
                if ((self.bot.entity.position.x - position.x) ** 2 + (self.bot.entity.position.z - position.z) ** 2) ** 0.5 > 7:
                    self.pathfinding(position)
                self.bot.chat(f"Attempting to place {name} at X: {position.x:.2f}, Y: {position.y:.2f}, Z: {position.z:.2f} using reference block in {direction} direction.")
                
                for attempt in range(3):  # Attempt to place the block up to 3 times
                    try:
                        self.equip(name, "hand")
                        self.moveToNearbyPosition(position)
                        self.bot.placeBlock(referenceBlock, faceVector)
                        self.bot.chat(f"The block {name} has been successfully placed at X: {position.x:.2f}, Y: {position.y:.2f}, Z: {position.z:.2f}.")
                        return  # Exit the retry loop if the block is placed successfully
                    except Exception as e:
                        if attempt < 2:  # If it's not the last attempt, notify and try again
                            self.bot.chat(" Retrying...")
                        else:  # On the last attempt, notify of the final failure
                            self.bot.chat(f"Final attempt failed to place {name} at X: {position.x:.2f}, Y: {position.y:.2f}, Z: {position.z:.2f}. Moving to the next reference block.")
                            print(str(e))
                        continue


            else:
                self.bot.chat(f"No valid reference block in {direction} direction, trying next reference block.")
        else:
            self.bot.chat(f"Failed to place {name} at X: {position.x:.2f}, Y: {position.y:.2f}, Z: {position.z:.2f}. Please check the surroundings and try again.")

        
        
    def jump_and_place_until(self, target_position, block_type):
        """
        Continuously makes the bot jump and place blocks under it until the vertical distance
        to the target position is less than 3 blocks.
        
        :param architectAgent: The agent controlling the bot.
        :param target_position: Vec3 representing the target position.
        :param block_type: The type of block to place (e.g., 'dirt').
        """
        while True:
            current_pos = self.bot.entity.position
            vertical_distance = target_position.y - current_pos.y

            print(f"Current Position: {current_pos}, Target Position: {target_position}")
            print(f"Vertical Distance: {vertical_distance}")

            if vertical_distance < 3:
                print("Reached the desired vertical proximity to the target position.")
                break

            # Check if any block above is not air
            blocks_above = [
                self.bot.blockAt(current_pos.offset(0, 1, 0)),
                self.bot.blockAt(current_pos.offset(0, 2, 0))
            ]

            for block in blocks_above:
                if block and block.name != 'air': 
                    print(f"Block above detected: {block.name}, digging...")
                    self.bot.dig(block)  
                    break 
            self.setInventorySlot(block_type, 64)
            self.jump_and_place(block_type)

            # Optional: Add a short delay to prevent rapid looping
            time.sleep(0.5)  # 500 milliseconds
    
    def jump_and_place(self, block_type):
        """Function to make the bot jump and place a block under it."""

        # Make the bot jump
        self.bot.setControlState('jump', True)
        # Simulate the jump duration
        time.sleep(0.1)  # 100 milliseconds
        self.bot.setControlState('jump', False)

        # Wait a short moment for the jump to initiate
        self.bot.waitForTicks(1)

        # Determine the reference block (block below the bot)
        current_pos = self.bot.entity.position
        target_pos = current_pos.offset(0, -1, 0)  # Position below the bot

        # Get the reference block at the target position
        reference_block = self.bot.blockAt(target_pos)

        if not reference_block:
            print("No block found to place against.")
            return

        # Determine the face to place the block (e.g., top face)
        face_vector = vec3(0, 1, 0)  # Top face

        # Calculate the position where the new block will be placed
        new_block_pos = vec3(
            reference_block.position.x + face_vector.x,
            reference_block.position.y + face_vector.y,
            reference_block.position.z + face_vector.z
        )

        # Place the block adjacent to the reference block
        try:
            # Assuming placeBlock is a synchronous function
            self.bot.placeBlock(reference_block, face_vector)
            print(f"Placed {block_type} at {new_block_pos}")
        except Exception as e:
            print(f"Failed to place block: {e}")

    
    
    def moveToNearbyPosition(self, targetPos):
        offsets = [
            vec3(4, 0, 0), vec3(-4, 0, 0),  # Check positions on the x-axis
            vec3(0, 0, 4), vec3(0, 0, -4),  # Check positions on the z-axis
            vec3(4, 0, 4), vec3(-4, 0, -4), # Check diagonal positions
            vec3(4, 0, -4), vec3(-4, 0, 4), # Check diagonal positions
        ]

        # Check if the bot is too close to the target position
        if self.bot.entity.position.distanceTo(targetPos) <= 2:
            for offset in offsets:
                check_pos = targetPos.plus(offset)
                block = self.bot.blockAt(check_pos)

                # Ensure the check position is not occupied and is not the target position
                if block and block.name == "air":
                    self.pathfinding(check_pos)
                    self.bot.chat("Moved to non-overlapping coordinate")
                    return

            self.bot.chat("Cannot find a non-overlapping coordinate nearby")
        else:
            self.bot.chat("Already far enough from the target position")



    def mineBlock(self,name,count=1):
        blockByName = self.bot.registry.blocksByName[name]
        blocks = self.bot.findBlocks({
                    "matching": blockByName.id,
                    "maxDistance": 70,
                    "count": count,
                    'timeout': 1000,
                })
        targets = [self.bot.blockAt(block) for block in blocks]
        self.bot.collectBlock.collect(targets, timeout=1000)
        self.bot.chat((f"I have finished mining {name}"))
    
    def craftItem(self, name, amount,useCraftingTable = True):
        # Retrieve the item by name
        item = self.bot.registry.itemsByName[name]
        
        if useCraftingTable:
            # Get the crafting table ID
            craftingTableID = self.bot.registry.blocksByName.crafting_table.id

            # Find the crafting table within a 32 block radius
            craftingTable = self.bot.findBlocks({
                "matching": craftingTableID,
                "maxDistance": 32,
            })

            # Pathfind to the first crafting table found
            self.pathfinding(craftingTable[0])

            # Get the block at the crafting table location
            craftingTableBlock = self.bot.blockAt(craftingTable[0])

            # Find the recipe for the item
            recipe = self.bot.recipesFor(item.id, None, amount, craftingTableBlock)[0]

            # Craft the item with a timeout of 1000ms
            self.bot.craft(recipe, amount, craftingTableBlock, timeout=1000)
        else:
            # Find the recipe for the item
            recipe = self.bot.recipesFor(item.id, None, amount, None)[0]

            # Craft the item with a timeout of 1000ms
            self.bot.craft(recipe, amount, None, timeout=1000)



    def smeltItems(self, itemName, fuelName, count):
        item = self.bot.registry.itemsByName[itemName]
        fuel = self.bot.registry.itemsByName[fuelName]
        furnaceBlock = self.bot.findBlock({
            'matching': self.bot.registry.blocksByName.furnace.id,
            'maxDistance': 32,
        })

        self.pathfinding(furnaceBlock.position)

        furnace = self.bot.openFurnace(furnaceBlock)
        for i in range(count):
            furnace.putFuel(fuel.id, None, 1)
            furnace.putInput(item.id, None, 1)
            # Wait 12 seconds for the furnace to smelt the item
            self.bot.waitForTicks(12 * 20)
            furnace.takeOutput()
        furnace.close()
    
    def observe(self, distance):
        entities = []
        # blocks = []
        
        # # Gather information about nearby blocks within the specified distance
        # for x in range(-distance, distance + 1):
        #     for y in range(-distance, distance + 1):
        #         for z in range(-distance, distance + 1):
        #             position = self.bot.entity.position.offset(x, y, z)
        #             block = self.bot.blockAt(position)
        #             if block and block.name != 'air':
        #                 blocks.append({
        #                     'name': block.name,
        #                     'position': block.position,
        #                     'type': block.type
        #                 })
        
        # Gather information about nearby entities within the specified distance
        for entityId in self.bot.entities:
            entity = self.bot.entities[entityId]
            if not entity or entity.position is None:
                continue  # Skip entities without a valid position

            entityPosition = entity.position
            botPosition = self.bot.entity.position
            distanceToEntity = math.sqrt(
                (entityPosition.x - botPosition.x) ** 2 +
                (entityPosition.y - botPosition.y) ** 2 +
                (entityPosition.z - botPosition.z) ** 2
            )

            if distanceToEntity <= distance:
                entities.append({
                    'name': entity.name,
                    'type': entity.kind,
                    'position': entityPosition
                })

        # return {'blocks': blocks, 'nearbyEntities': entities}
        return { 'nearbyEntities': entities}
    
    
    def getState(self):
        # Get the bot's health
        health = self.bot.health
        
        # Get the bot's food level
        food = self.bot.food
        
        # Get the bot's inventory items
        inventory = []
        for slot in self.bot.inventory.slots:
            if slot is not None:  # Only process non-empty slots
                inventory.append({
                    'name': slot.name,
                    'count': slot.count,
                    'slot': slot.slot
                })

        return {'health': health, 'food': food, 'inventory': inventory}
    
    def moveToChest(self, chestPosition=None): 
        item = self.bot.registry.itemsByName['chest']
        chestBlock = self.bot.findBlock({
            'matching': self.bot.registry.blocksByName.chest.id,
            'maxDistance': 32,
        })

        self.pathfinding(chestBlock.position)
        if chestPosition:
            self.pathfinding(chestPosition)
            chestBlock = self.bot.blockAt(chestPosition)
        if chestBlock.name != 'chest':
            self.log(f"No chest at {chestPosition}, it is {chestBlock.name}")
            return None
        return chestBlock
        
    def listItemsInChest(self, chestBlock): 
        chest = self.bot.openContainer(chestBlock)
        items = chest.containerItems()
        itemNames = {}
        if items: 
            for item in items: 
                if item.name in itemNames: 
                    itemNames[item.name] += item.count
                else: 
                    itemNames[item.name] = item.count
        self.bot.emit("closeChest", itemNames, chestBlock.position)
        print(itemNames)
        chest.close()
        return chest 
    
    def depositItemIntoChest(self, chestBlock, itemsToDeposit): 
        chest = self.bot.openContainer(chestBlock)
        for name, quantity in itemsToDeposit.items(): 
            itemByName = self.bot.registry.itemsByName[name]
            if not itemByName: 
                self.bot.chat(f"No item named {name}")
                continue
            item = self.bot.inventory.findInventoryItem(itemByName.id)
            if not item: 
                self.bot.chat(f"No {name} in inventory")
                continue
            try: 
                chest.deposit(item.type, None, quantity)
            except Exception as err: 
                self.bot.chat(f"Not enough {name} in inventory")
        chest.close()
        return chest
    
    def getItemsFromChest(self, chestBlock, itemsToGet):
        chest = self.bot.openContainer(chestBlock)
        for name, quantity in itemsToGet.items():
            itemByName = self.bot.registry.itemsByName[nade]
            if not itemByName:
                self.bot.chat(f"No item named {name}")
                continue

            item = chest.findContainerItem(itemByName.id)
            if not item:
                self.bot.chat(f"I don't see {name} in this chest")
                continue

            try:
                chest.withdraw(item.type, None, quantity)
            except Exception as err:
                self.bot.chat(f"Not enough {name} in chest")
        chest.close()
        return chest
    

    def findNearestEntity(self, entityName=None, maxDistance=float('inf')):
        nearest_entity = None
        nearest_distance = maxDistance

        for entity_id in self.bot.entities:
            entity = self.bot.entities[entity_id]
            if not entity or entity.position is None:
                continue  # Skip entities without a valid position

            if entityName and entity.name.lower() != entityName.lower():
                continue  # Skip entities that don't match the specified name

            entity_position = entity.position
            bot_position = self.bot.entity.position

            distance_to_entity = math.sqrt(
                (entity_position.x - bot_position.x) ** 2 +
                (entity_position.y - bot_position.y) ** 2 +
                (entity_position.z - bot_position.z) ** 2
            )

            if distance_to_entity <= nearest_distance:
                nearest_entity = entity
                nearest_distance = distance_to_entity

        return nearest_entity if nearest_entity else None

    def killMob(self,mobName):
        entity = self.findNearestEntity(entityName=mobName, maxDistance=64)
        if entity:
            self.pathfinding(vec3(entity.position.x,entity.position.y,entity.position.z))
            self.bot.pvp.attack(entity)
            self.bot.chat(f"killed {mobName}")
        else:
            self.bot.chat(f"Did not find {mobName}")
    

    def getPosition(self):
        return self.bot.entity.position
    

    def buildStructure(self, architecture_layout, mode = 'creative'):
        if mode == 'creative':
            for blockType, pos in architecture_layout:
                # Set the inventory slot to contain the required block type
                self.setInventorySlot(blockType, 64)
                
                # Place the block at the specified position
                self.placeBlockAt(blockType, pos)
        
        if mode == 'survival':
            for blockType, pos in architecture_layout:       
                # Place the block at the specified position
                self.placeBlockAt(blockType, pos)

        self.bot.chat("Structure has been built according to the provided layout.")