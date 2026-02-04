#!/bin/bash
set -e
set -u

function create_user_and_database() {
	local database=$1
	echo "  Checking database '$database'"
	
	# Check if DB exists
	if psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" -lqt | cut -d \| -f 1 | grep -qw $database; then
	    echo "  Database '$database' already exists. Skipping."
	else
	    echo "  Creating database '$database'"
	    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
	        CREATE DATABASE $database;
	        GRANT ALL PRIVILEGES ON DATABASE $database TO $POSTGRES_USER;
EOSQL
	fi
}

if [ -n "$POSTGRES_MULTIPLE_DATABASES" ]; then
	echo "Multiple database creation requested: $POSTGRES_MULTIPLE_DATABASES"
	for db in $(echo $POSTGRES_MULTIPLE_DATABASES | tr ',' ' '); do
		create_user_and_database $db
	done
	echo "Multiple databases setup complete."
fi